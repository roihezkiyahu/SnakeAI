from torch.nn import utils
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modeling.trainer import Trainer
from SnakeGame.snake_game import SnakeGame
from modeling.models import ActorCritic
import matplotlib
from torch.distributions import Categorical
try:
    matplotlib.use('Agg')
except:
    print("no TkAgg")
import os
import random
import yaml
from modeling.trainer import Debugger

class A2CDebugger():
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
        actor_gradients, critic_gradients = [], []
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

    def plot_loss(self, epochs, window=100):
        actor_losses, value_losses = zip(*self.loss_history)
        plt.subplot(221)
        act_running_avg = Debugger.moving_average(actor_losses, window)
        crit_running_avg = Debugger.moving_average(value_losses, window)
        n_epochs = len(epochs)
        plt.plot(epochs, actor_losses, label='Actor Loss')
        plt.plot(range(window, n_epochs + 1), act_running_avg, label=f'Actor Running Average {window}',
                 linestyle='dashed')
        plt.plot(epochs, value_losses, label='Critic Loss')
        plt.plot(range(window, n_epochs + 1), crit_running_avg, label=f'Critic Running Average {window}',
                 linestyle='dashed')
        plt.title('Losses over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

    def plot_grads(self, epochs, window=100):
        plt.subplot(222)
        actor_grads, critic_grads = self.gradient_norms_actor, self.gradient_norms_critic
        act_running_avg = Debugger.moving_average(actor_grads, window)
        crit_running_avg = Debugger.moving_average(critic_grads, window)
        n_epochs = len(epochs)
        plt.plot(epochs, actor_grads, label='Actor Gradient Norms')
        plt.plot(range(window, n_epochs + 1), act_running_avg, label=f'Actor Running Average {window}',
                 linestyle='dashed')
        plt.plot(epochs, critic_grads, label='Critic Gradient Norms')
        plt.plot(range(window, n_epochs + 1), crit_running_avg, label=f'Critic Running Average {window}',
                 linestyle='dashed')
        plt.title('Gradient Norms over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend()

    def plot_entropy(self, epochs):
        plt.subplot(223)
        plt.plot(epochs, self.policy_entropy)
        plt.title('Policy Entropy')
        plt.xlabel('Epoch')
        plt.ylabel('Entropy')

    def plot_scores(self, window):
        plt.subplot(224)
        scores = np.array(self.score_history)
        n_scores = len(scores)
        window = min(window, n_scores)
        running_avg = Debugger.moving_average(scores, window)
        running_avg_3 = Debugger.moving_average(scores, min(window * 3, n_scores))
        running_avg_5 = Debugger.moving_average(scores, min(window * 5, n_scores))
        plt.plot(range(1, n_scores + 1), scores, label='Score')
        plt.plot(range(window, n_scores + 1), running_avg, label=f'Running Average {window}', linestyle='dashed')
        plt.plot(range(window * 3, n_scores + 1), running_avg_3, label=f'Running Average {window * 3}', linestyle='dashed')
        plt.plot(range(window * 5, n_scores + 1), running_avg_5, label=f'Running Average {window * 5}', linestyle='dashed')
        plt.title('Rewards History')
        plt.xlabel('Game Number')
        plt.ylabel('Score')
        plt.legend()

    def plot_diagnostics(self, epoch, window=100):
        epochs = range(len(self.loss_history))
        plt.figure(figsize=(15, 10))
        self.plot_loss(epochs)
        self.plot_grads(epochs)
        self.plot_entropy(epochs)
        self.plot_scores(window)
        plt.tight_layout()
        filename = f"{self.agent.prefix_name}_{epoch}_diagnostics.png"
        plt.savefig(filename)
        print(f"Saved diagnostics to {filename}")
        plt.close()


class A2CAgent(Trainer):
    def __init__(self, config_path, value_network, actor_network):
        super().__init__(config_path, value_network, actor_network)
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)['trainer']
        else:
            config = config_path
        self.value_network = value_network.to(self.device)
        self.actor_network = actor_network.to(self.device)
        self.value_optimizer = optim.RMSprop(self.value_network.parameters(), lr=float(config["value_network_lr"]))
        self.actor_optimizer = optim.RMSprop(self.actor_network.parameters(), lr=float(config["actor_network_lr"]))
        self.entropy_coefficient = float(config["entropy_coefficient"])
        input_shape = config["input_shape"]
        self.input_shape = eval(input_shape) if isinstance(input_shape, str) else input_shape
        self.debugger = A2CDebugger(self)

    def check_early_stop(self, episode_count):
        if (episode_count+1) % self.validate_every_n_episodes == 0:
            if self.early_stopping:
                mean_scores = np.array([val_log["Mean Score"] for val_log in self.validation_log])
                if not len(mean_scores):
                    return False
                if np.all(max(mean_scores) > mean_scores[-min(self.early_stopping, len(mean_scores)):]):
                    print("early stopped")
                    return True
        return False

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=0)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self, epochs, batch_size):
        actions, dones, rewards, values, observations = self.initialize_batch_variables(batch_size)
        obs, info = self.game_wrapper.reset(self.reset_options)
        episode_count, total_reward, steps = 0, 0, 1
        for epoch in range(epochs):
            i = 0
            while i < batch_size:
                action, policy, value, obs_torch = self.get_action_and_value(obs)
                actions[i] = action
                values[i] = value
                observations[i] = obs_torch
                obs, reward, done = self.step_game(action)
                rewards[i], dones[i], total_reward = reward, done, total_reward + reward
                i, steps, episode_count, total_reward, obs = self.handle_episode_end(i, done, steps, total_reward,
                                                                                     episode_count, obs, policy, action)
            if self.check_early_stop(episode_count):
                break
            self.update_model(observations, actions, rewards, dones, values, obs_torch)
            self.save_diagnostics_if_needed(epoch)
        print(f'The training was done over a total of {episode_count} episodes')

    def initialize_batch_variables(self, batch_size):
        actions, dones = np.zeros((batch_size,), dtype=np.int32), np.zeros((batch_size,), dtype=bool)
        rewards, values = np.zeros((2, batch_size), dtype=np.float32)
        observations = torch.zeros((batch_size,) + self.input_shape, dtype=torch.float32).to(self.device)
        return actions, dones, rewards, values, observations

    def get_action_and_value(self, obs):
        obs_torch = torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(self.device)
        value = self.value_network(obs_torch).cpu().detach().numpy()
        policy = self.actor_network(obs_torch)
        action = Categorical(logits=policy).sample().cpu().detach().numpy()[0]
        return action, policy, value, obs_torch

    def step_game(self, action):
        obs, reward, terminated, truncated, _ = self.game_wrapper.step(action)
        done = terminated or truncated
        return obs, reward, done

    def handle_episode_end(self, i, done, steps, total_reward, episode_count, obs, policy, action):
        if (episode_count + 1) % self.save_gif_every_x_epochs == 0:
            probs = torch.round(F.softmax(policy[0], dim=-1) * 100).cpu().int().tolist()
            self.visualize_and_save_game_state(episode_count, self.game_wrapper.preprocessor.postprocess_action(action),
                                               probs)
        if done:
            self.log_episode(episode_count, total_reward)
            total_reward, steps, episode_count = 0, -1, episode_count + 1
            obs, info = self.game_wrapper.reset(self.reset_options)
        return i + 1, steps + 1, episode_count, total_reward, obs

    def log_episode(self, episode_count, total_reward):
        score = self.game_wrapper.get_score()
        print(" " * 100, end="\r")
        print(f"episode: {episode_count}, reward: {total_reward}, score: {score}", end="\r")
        self.rewards_memory.append(total_reward)
        self.score_memory.append(score)
        self.log_and_compile_gif(episode_count)
        if (episode_count + 1) % self.validate_every_n_episodes == 0:
            self.model = self.actor_network
            self.validate_score(episode_count)
        self.debugger.track_scores([score])

    def update_model(self, observations, actions, rewards, dones, values, obs_torch):
        next_value = [0] if dones[-1] else self.value_network(obs_torch).cpu().detach().numpy()[0]
        returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
        self.optimize_model(observations, actions, returns, advantages)

    def save_diagnostics_if_needed(self, epoch):
        if (epoch + 1) % self.save_diagnostics == 0:
            self.debugger.plot_diagnostics(epoch + 1)
            print(f"saved diagnostics epoch: {epoch + 1}")

    def optimize_model(self, observations, actions, returns, advantages):
        actions, returns, advantages, observations = self.prepare_tensors(actions, returns, advantages, observations)
        value_loss = self.optimize_value_network(observations, returns)
        actor_loss, entropy = self.optimize_actor_network(observations, actions, advantages)
        self.debugger.track_loss(actor_loss, value_loss)
        self.debugger.track_policy_entropy(entropy.cpu().detach().numpy())

    def prepare_tensors(self, actions, returns, advantages, observations):
        actions = F.one_hot(torch.tensor(actions, dtype=torch.int64), self.n_actions).float().to(self.device)
        returns = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        observations = torch.tensor(observations, dtype=torch.float).to(self.device)
        return actions, returns, advantages, observations

    def optimize_value_network(self, observations, returns):
        self.value_optimizer.zero_grad()
        values = self.value_network(observations)
        value_loss = (0.5 * (values - returns) ** 2).mean()
        value_loss.backward()
        self.debugger.track_gradients()
        self.apply_grad_clip()
        self.value_optimizer.step()
        return value_loss

    def optimize_actor_network(self, observations, actions, advantages):
        self.actor_optimizer.zero_grad()
        policies = self.actor_network(observations)
        actor_loss, entropy = self.compute_actor_loss_entropy(policies, actions, advantages)
        actor_loss.backward()
        self.debugger.track_gradients(True)
        self.apply_grad_clip(value=False)
        self.actor_optimizer.step()
        return actor_loss, entropy

    def apply_grad_clip(self, value=True):
        if self.clip_grad > 0:
            if value:
                utils.clip_grad_norm_(self.value_network.parameters(), self.clip_grad)
            else:
                utils.clip_grad_norm_(self.actor_network.parameters(), self.clip_grad)

    def compute_actor_loss_entropy(self, policies, actions, advantages):
        probs, log_probs, log_action_probs = self.compute_probabilities(policies, actions)
        actor_loss = -(log_action_probs * advantages).mean()
        entropy = -(probs * log_probs).sum(-1).mean()
        actor_loss -= self.entropy_coefficient * entropy
        return actor_loss, entropy

    @staticmethod
    def compute_probabilities(policies, actions):
        probs = F.softmax(policies + 1e-9, dim=-1)
        log_probs = F.log_softmax(policies + 1e-9, dim=-1)
        log_action_probs = torch.sum(log_probs * actions, dim=1)
        return probs, log_probs, log_action_probs


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
