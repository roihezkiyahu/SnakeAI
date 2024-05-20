from torch.nn import utils
import torch
import torch.optim as optim
import numpy as np
from processing import preprocess_state, postprocess_action
import torch.nn.functional as F
import matplotlib.pyplot as plt
from trainer import Trainer
from SnakeGame.snake_game import SnakeGame
from models import ActorCritic
import matplotlib
from torch.distributions import Categorical
try:
    matplotlib.use('TkAgg')
except:
    print("no TkAgg")
import os


class A2CDebugger:
    def __init__(self, agent):
        self.agent = agent
        self.loss_history = []
        self.gradient_norms_actor = []
        self.gradient_norms_critic = []
        self.policy_entropy = []
        self.score_history = []
        self.value_outputs = []

    def track_loss(self, actor_loss, value_loss):
        self.loss_history.append((actor_loss.item(), value_loss.item()))

    def track_gradients(self, actor=False):
        actor_gradients = []
        critic_gradients = []
        for p in self.agent.actor_network.parameters():
            if p.grad is not None:
                actor_gradients.append(p.grad.norm().item())
        for p in self.agent.value_network.parameters():
            if p.grad is not None:
                critic_gradients.append(p.grad.norm().item())
        if actor:
            self.gradient_norms_actor.append(np.mean(actor_gradients))
        else:
            self.gradient_norms_critic.append(np.mean(critic_gradients))

    def track_policy_entropy(self, entropy):
        self.policy_entropy.append(entropy)

    def track_scores(self, score):
        self.score_history.extend(score)

    def track_value_outputs(self):
        with torch.no_grad():
            for observation in self.agent.env.observations:
                value = self.agent.value_network(torch.tensor(observation, dtype=torch.float)).item()
                self.value_outputs.append(value)

    def plot_diagnostics(self, epoch):
        epochs = range(len(self.loss_history))
        actor_losses, value_losses = zip(*self.loss_history)

        plt.figure(figsize=(15, 10))

        plt.subplot(221)
        plt.plot(epochs, actor_losses, label='Actor Loss')
        plt.plot(epochs, value_losses, label='Critic Loss')
        plt.title('Losses over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(222)
        actor_grads, critic_grads = self.gradient_norms_actor, self.gradient_norms_critic
        plt.plot(epochs, actor_grads, label='Actor Gradient Norms')
        plt.plot(epochs, critic_grads, label='Critic Gradient Norms')
        plt.title('Gradient Norms over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend()

        plt.subplot(223)
        plt.plot(epochs, self.policy_entropy)
        plt.title('Policy Entropy')
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')

        plt.subplot(224)
        plt.plot(range(1, len(self.score_history) + 1), self.score_history)
        plt.title('Rewards History')
        plt.xlabel('Game Number')
        plt.ylabel('Score')

        plt.tight_layout()

        # Save the figure
        filename = f"{self.agent.prefix_name}_{epoch}_diagnostics.png"
        plt.savefig(filename)
        print(f"Saved diagnostics to {filename}")

        plt.close()

class A2CAgent(Trainer):
    def __init__(self, game, value_network, actor_network, value_network_lr=1e-4, actor_network_lr=1e-4,
                 gamma=0.99,
                 reward_params={'death': 0, 'move': 0, 'food': 0, "food_length_dependent": 1,
                                "death_length_dependent": -1},
                 episodes=10000,
                 learning_rate=5e-5,
                 n_memory_episodes=100,
                 prefix_name="",
                 folder="",
                 save_gif_every_x_epochs=500,
                 EPS_START=1,
                 EPS_END=0,
                 EPS_DECAY=250,
                 max_episode_len=10000,
                 close_food=2500,
                 close_food_episodes_skip=100,
                 max_init_len=5,
                 replaymemory=10000,
                 discount_rate=0.99,
                 per_alpha=0.6,
                 validate_every_n_episodes=500,
                 validate_episodes=100,
                 increasing_start_len=False,
                 patience=3,
                 entropy_coefficient=0.01,
                 input_shape=(11, 12, 12),
                 clip_grad=0,
                 save_diagnostics=2500
                 ):
        super().__init__(game, value_network, actor_network, gamma=gamma, reward_params=reward_params,
                         max_init_len=max_init_len, close_food=close_food,
                         close_food_episodes_skip=close_food_episodes_skip, increasing_start_len=increasing_start_len,
                         n_memory_episodes=n_memory_episodes, prefix_name=prefix_name, folder=folder,
                         save_gif_every_x_epochs=save_gif_every_x_epochs,
                         max_episode_len=max_episode_len, discount_rate=discount_rate,
                         validate_every_n_episodes=validate_every_n_episodes, validate_episodes=validate_episodes)
        self.value_network = value_network.to(self.device)
        self.actor_network = actor_network.to(self.device)
        self.value_optimizer = optim.RMSprop(self.value_network.parameters(), lr=value_network_lr)
        self.actor_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=actor_network_lr)
        self.entropy_coefficient = entropy_coefficient
        self.input_shape = input_shape
        self.debugger = A2CDebugger(self)
        self.clip_grad = clip_grad
        self.save_diagnostics = save_diagnostics

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=0)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self, epochs, batch_size):
        episode_count = 0
        actions = np.zeros((batch_size,), dtype=np.int32)
        dones = np.zeros((batch_size,), dtype=bool)
        rewards, values = np.zeros((2, batch_size), dtype=np.float32)
        observations = np.zeros((batch_size,) + self.input_shape, dtype=np.float32)
        obs = self.init_episode(episode_count)
        last_action = self.game.snake_direction
        total_reward, last_score, steps = 0, 0, 1


        for epoch in range(epochs):
            for i in range(batch_size):
                observations[i] = obs
                obs_torch = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)
                values[i] = self.value_network(obs_torch).cpu().detach().numpy()
                policy = self.actor_network(obs_torch)
                actions[i] = Categorical(logits=policy).sample().cpu().detach().numpy()
                game_action = postprocess_action(actions[i])
                self.game.change_direction(game_action)
                score, done = self.game.move()
                if self.check_failed_init(steps, done, epoch, game_action, policy, last_action):
                    rewards[i] = 0
                    obs = self.init_episode(episode_count)
                    continue
                reward = self.compute_reward(score, last_score, done, last_action != game_action, len(self.game.snake))
                last_score = score
                total_reward += reward
                obs, rewards[i], dones[i]= preprocess_state(self.game), reward, done

                if (episode_count + 1) % self.save_gif_every_x_epochs == 0:
                    probs = torch.round(F.softmax(policy[0], dim=-1) * 100).cpu().int().tolist()
                    self.visualize_and_save_game_state(episode_count, game_action, probs)
                if dones[i]:
                    obs = self.init_episode(episode_count)

                steps +=1
                if steps >= self.max_episode_len:
                    done = True
                    obs = self.init_episode(episode_count)
                if done:
                    print(" " * 100, end="\r")
                    print(f"episode: {episode_count}, reward: {total_reward}, score: {score}", end="\r")
                    self.rewards_memory.append(total_reward)
                    self.score_memory.append(score)
                    self.log_and_compile_gif(episode_count)
                    if (episode_count+1) % self.validate_every_n_episodes == 0 or epoch == epochs - 1:
                        self.model = self.actor_network
                        self.validate_score(episode_count)
                    episode_count += 1
                    self.debugger.track_scores([score])
                    total_reward = 0
                    steps = 0
                    last_score = 0

            if dones[-1]:
                next_value = [0]
            else:
                next_value = self.value_network(obs_torch).cpu().detach().numpy()[0]
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
            self.optimize_model(observations, actions, returns, advantages)

            if (epoch+1) % self.save_diagnostics == 0:
                self.debugger.plot_diagnostics(epoch+1)
                print(f"saved diagnostics epoch: {epoch+1}")

        print(f'The trainnig was done over a total of {episode_count} episodes')

    def optimize_model(self, observations, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions, dtype=torch.int64), 4).float().to(self.device)
        returns = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        observations = torch.tensor(observations, dtype=torch.float).to(self.device)

        self.value_optimizer.zero_grad()
        values = self.value_network(observations)
        value_loss = (0.5*(values-returns)**2).mean()
        value_loss.backward()
        self.debugger.track_gradients()
        if self.clip_grad > 0:
            utils.clip_grad_norm_(self.value_network.parameters(), self.clip_grad)
        self.value_optimizer.step()

        self.actor_optimizer.zero_grad()
        policies = self.actor_network(observations)
        probs = F.softmax(policies +1e-9, dim=-1)
        log_probs = F.log_softmax(policies+1e-9, dim=-1)
        log_action_probs = torch.sum(log_probs*actions, dim=1)
        actor_loss = -(log_action_probs * advantages).mean()

        entropy = -(probs * log_probs).sum(-1).mean()
        actor_loss -= self.entropy_coefficient * entropy

        actor_loss.backward()
        self.debugger.track_gradients(True)
        if self.clip_grad > 0:
            utils.clip_grad_norm_(self.actor_network.parameters(), self.clip_grad)
        self.actor_optimizer.step()

        self.debugger.track_loss(actor_loss, value_loss)
        self.debugger.track_policy_entropy(entropy.cpu().detach().numpy())

if __name__ == "__main__":
    conv_layers_params = [
        {'in_channels': 11, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ]
    fc_layers = [256]

    action_size = 4
    game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    input_shape = (11, game.width + 2, game.height + 2)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad5_bs256_close5000_20k",
    #                value_network_lr=1e-4, actor_network_lr=1e-4,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=500,
    #                increasing_start_len=True, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=5)
    #
    # A2C.training_batch(20000, 256)


    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad1_bs128_5e5_close5000_2",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=12500,
    #                increasing_start_len=False, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=1)
    #
    # A2C.training_batch(40000, 128)


    game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')

    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad1_bs128_5e5_close5000_entroy05",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=12500,
    #                increasing_start_len=False, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=1, entropy_coefficient=0.05)
    #
    # A2C.training_batch(40000, 128)




    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad5_bs512_1e5_close5000",
    #                value_network_lr=1e-5, actor_network_lr=1e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=12500,
    #                increasing_start_len=False, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=5)
    #
    # A2C.training_batch(40000, 512)


    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad1_bs128_5e5_close5000_inc_start_len",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=12500,
    #                increasing_start_len=True, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=1)
    #
    # A2C.training_batch(40000, 128)


    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad1_bs64_5e5_close5000",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=12500,
    #                increasing_start_len=True, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=1)
    #
    # A2C.training_batch(20000, 64)

    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad100_bs128_5e5_close5000",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=12500,
    #                increasing_start_len=False, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=100)
    #
    # A2C.training_batch(60000, 128)


    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor')
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic')
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2C_clip_grad1_bs128_5e5_close5000_entroy005",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=12500,
    #                increasing_start_len=False, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=1, entropy_coefficient=0.005)
    #
    # A2C.training_batch(60000, 128)

    # conv_layers_params = [
    #     {'in_channels': 11, 'out_channels': 11, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 11},
    #     {'in_channels': 11, 'out_channels': 16, 'kernel_size': 1, 'stride': 1},
    #     {'in_channels': 16, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    #
    #     {'in_channels': 16, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 16},
    #     {'in_channels': 16, 'out_channels': 16, 'kernel_size': 1, 'stride': 1},
    #     {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    #
    #     {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 32},
    #     {'in_channels': 32, 'out_channels': 32, 'kernel_size': 1, 'stride': 1},
    #     {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    # ]
    # fc_layers = [256]
    #
    # action_size = 4
    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # input_shape = (11, game.width + 2, game.height + 2)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_clip_grad100_bs128_close5k_60k_seperable",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=500,
    #                increasing_start_len=True, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=100)
    #
    # A2C.training_batch(60000, 128)

    conv_layers_params = [
        {'in_channels': 11, 'out_channels': 11, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 11},
        {'in_channels': 11, 'out_channels': 16, 'kernel_size': 1, 'stride': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ]
    fc_layers = [256]


    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_clip_grad100_bs128_inclen",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=500,
    #                increasing_start_len=True, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=100)
    #
    # A2C.training_batch(100000, 128)


    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_clip_grad100_bs256_inclen_nolendep",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': -1.5, 'move': 0, 'food': 1,
    #                               "food_length_dependent": 0, "death_length_dependent": 0},
    #                validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=2500,
    #                increasing_start_len=True, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=100)
    #
    # A2C.training_batch(100000, 256)


    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_clip_grad100_bs128_entroy005_max_init50_depthconv",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=500,
    #                max_init_len=50, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=100, entropy_coefficient=0.005)
    #
    # A2C.training_batch(60000, 128)
    #
    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_clip_grad100_bs128_entroy005_depthconv",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 1, "death_length_dependent": -1},
    #                validate_every_n_episodes=1000, validate_episodes=100, save_gif_every_x_epochs=500, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250,
    #                clip_grad=100, entropy_coefficient=0.005)
    #
    # A2C.training_batch(60000, 128)

    # game = SnakeGame(10, 10, 10, default_start_prob=0.25)
    # actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_bs512_inclen",
    #                value_network_lr=5e-5, actor_network_lr=5e-5,
    #                close_food=5000, close_food_episodes_skip=500,
    #                reward_params={'death': 0, 'move': 0, 'food': 0,
    #                               "food_length_dependent": 0.5, "death_length_dependent": -0.6},
    #                validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=2500,
    #                increasing_start_len=True, episodes=50000,
    #                max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250)
    #
    # A2C.training_batch(100000, 512)

    conv_layers_params = [
        {'in_channels': 11, 'out_channels': 11, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups':11},
        {'in_channels': 11, 'out_channels': 16, 'kernel_size': 1, 'stride': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ]
    fc_layers = [256]
    game = SnakeGame(10, 10, 10, default_start_prob=0.1)

    input_shape = (11, game.width + 2, game.height + 2)
    action_size = 4
    actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
                              use_instance_norm=True)
    critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
                               use_instance_norm=True)

    A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_bs256_inclen_nolendep_morechannels_dsp01_2",
                   value_network_lr=5e-5, actor_network_lr=5e-5,
                   close_food=5000, close_food_episodes_skip=500,
                   reward_params={'death': -1.5, 'move': 0, 'food': 1,
                                  "food_length_dependent": 0, "death_length_dependent": 0},
                   validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=2500,
                   increasing_start_len=True, episodes=50000,
                   max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250)

    A2C.training_batch(150000, 256)


    game = SnakeGame(10, 10, 10, default_start_prob=0.1)

    input_shape = (11, game.width + 2, game.height + 2)
    action_size = 4
    actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
                              use_instance_norm=True)
    critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
                               use_instance_norm=True)

    A2C = A2CAgent(game, critic_model, actor_model, folder="A2CIN_bs256_inclen_nolendep_morechannels_dsp01_lr1e5",
                   value_network_lr=1e-5, actor_network_lr=1e-5,
                   close_food=5000, close_food_episodes_skip=500,
                   reward_params={'death': -1.5, 'move': 0, 'food': 1,
                                  "food_length_dependent": 0, "death_length_dependent": 0},
                   validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=2500,
                   increasing_start_len=True, episodes=50000,
                   max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250)

    A2C.training_batch(150000, 256)
