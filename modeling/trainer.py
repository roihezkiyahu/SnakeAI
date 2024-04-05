from game_viz import GameVisualizer
import torch
import torch.nn as nn
import torch.optim as optim
import os
from memory import ReplayMemory, Transition, PERMemory
import math
import numpy as np
from processing import preprocess_state, postprocess_action
import torch.nn.functional as F
import random
import imageio
import matplotlib.pyplot as plt

class Trainer:
    def __init__(
            self,
            game,
            model,
            clone_model,
            episodes=10000,
            learning_rate=5e-5,
            gamma=0.99,
            reward_params={'death': 0, 'move': 0, 'food': 0, "food_length_dependent": 1, "death_length_dependent": -1},
            n_memory_episodes=100,
            prefix_name="",
            folder="",
            save_gif_every_x_epochs=500,
            batch_size=1024,
            EPS_START=1,
            EPS_END=0,
            EPS_DECAY=250,
            TAU=5e-3,
            max_episode_len=10000,
            close_food=2500,
            close_food_episodes_skip=100,
            use_ddqn=True,
            max_init_len=15,
            replaymemory=10000,
            optimizer=None,
            discount_rate=0.99,
            per_alpha=0.6,
            use_scheduler=False,
            validate_every_n_episodes=500,
            validate_episodes=100,
            increasing_start_len=False,
    ):
        self.use_ddqn = use_ddqn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = game
        self.visualizer = GameVisualizer(game)
        self.model = model.to(self.device)
        self.target_net = clone_model.to(self.device)
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reward_params = reward_params
        self.n_memory_episodes = n_memory_episodes
        self.batch_size = batch_size
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.max_episode_len = max_episode_len
        self.close_food = close_food
        self.close_food_episodes_skip = close_food_episodes_skip
        self.save_gif_every_x_epochs = save_gif_every_x_epochs

        if folder:
            os.makedirs(folder, exist_ok=True)
            self.prefix_name = os.path.join(folder, prefix_name)
        else:
            self.prefix_name = prefix_name

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) if not optimizer else optimizer
        self.use_scheduler = use_scheduler
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                        patience=1, threshold=0.01, verbose=True)

        self.criterion = nn.MSELoss()
        self.rewards_memory = []
        self.score_memory = []
        self.frames = []
        self.memory = PERMemory(replaymemory, per_alpha)
        self.max_init_len = max_init_len
        self.discount_rate = discount_rate
        self.validate_every_n_episodes = validate_every_n_episodes
        self.validate_episodes = validate_episodes
        self.increasing_start_len = increasing_start_len

    def choose_action(self, state, episode, validation=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if validation:
            with torch.no_grad():
                probs = self.model(state_tensor)
                return torch.argmax(probs).item(), torch.round(F.softmax(probs[0]) * 100).cpu().int().tolist()
        else:
            epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * episode / self.EPS_DECAY)
            if torch.rand(1, device=self.device).item() < epsilon:
                return torch.randint(0, 4, (1,), device=self.device).item(), [0, 0, 0, 0]
            else:
                with torch.no_grad():
                    probs = self.model(state_tensor)
                    return torch.argmax(probs).item(), torch.round(F.softmax(probs[0]) * 100).cpu().int().tolist()

    def compute_loss(self, log_probs, rewards):
        discounted_rewards = self.discount_rewards(rewards, self.discount_rate)
        loss = -torch.sum(torch.stack(log_probs) * torch.FloatTensor(discounted_rewards).to(self.device))
        return loss

    def discount_rewards(self, rewards):
        discounted = np.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.discount_rate * R
            discounted[t] = R
        return discounted

    def compute_reward(self, score, last_score, done, move, snake_len):
        reward = self.reward_params['death'] * done
        reward += self.reward_params['move'] * move
        reward += (score - last_score) * self.reward_params['food']
        reward += (score - last_score) * self.reward_params.get('food_length_dependent', 0) * snake_len
        reward += done * self.reward_params.get('death_length_dependent', 0) * snake_len
        return reward

    def update_state(self, done):
        if done:
            return None, None
        next_state = preprocess_state(self.game)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        return next_state, next_state_tensor

    def get_batch(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool,
                                      device=self.device)

        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        return state_batch, action_batch, reward_batch, next_state_batch, non_final_mask, weights, indices

    def get_state_target_value(self, state_batch, action_batch, non_final_next_states, non_final_mask):
        state_action_values = self.model(state_batch).gather(1, action_batch)
        target_net_pred = self.target_net(non_final_next_states)
        if self.use_ddqn:
            best_actions = self.model(non_final_next_states).max(1)[1].unsqueeze(1)
            target_action_values = torch.zeros(self.batch_size, device=self.device)
            target_action_values[non_final_mask] = target_net_pred.gather(1, best_actions).squeeze()
        else:
            target_action_values = torch.zeros(self.batch_size, device=self.device)
            target_action_values[non_final_mask] = target_net_pred.max(1)[0]
        return state_action_values, target_action_values

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, non_final_mask, weights, indices = self.get_batch()
        state_action_values = self.model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = (weights * F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1),
                                           reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).abs().squeeze().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

    def init_game_max_food_distance(self, episode):
        if episode > self.close_food:
            self.game.max_food_distance = None
        else:
            self.game.max_food_distance = episode // self.close_food_episodes_skip + 1

    def check_failed_init(self, steps, done, episode, game_action, probs, last_action, debug=False):
        if steps <= 1 and done:
            if debug:
                print("failed attempt")
                image = self.visualizer.save_current_frame(game_action, probs)
                plt.imshow(image)
                plt.show()
                print(f"{last_action}, {game_action}")
            if (episode + 1) % self.save_gif_every_x_epochs == 0:
                self.visualize_and_save_game_state(episode, game_action, probs)
            return True
        return False

    def run_episode(self, episode):
        self.init_game_max_food_distance(episode)
        self.game.reset_game(random.randint(2, self.max_init_len))
        state = preprocess_state(self.game)
        done, last_score, steps, score, rewards, last_action = False, 0, 0, np.nan, [], self.game.snake_direction

        while not done and steps <= self.max_episode_len:
            steps += 1
            action, probs = self.choose_action(state, episode)
            game_action = postprocess_action(action)
            self.game.change_direction(game_action)
            score, done = self.game.move()
            if self.check_failed_init(steps, done, episode, game_action, probs, last_action):
                rewards.append(np.nan)
                break

            reward = self.compute_reward(score, last_score, done, last_action != game_action, len(self.game.snake))
            last_score = score
            rewards.append(reward)

            next_state, next_state_tensor = self.update_state(done)
            self.memory.push(torch.tensor(state, device=self.device),
                             torch.tensor([action], device=self.device),
                             next_state_tensor, torch.tensor([reward], device=self.device))
            state = next_state
            self.optimize_model()

            target_net_state_dict, policy_net_state_dict = self.target_net.state_dict(), self.model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
            self.target_net.load_state_dict(target_net_state_dict)

            if (episode + 1) % self.save_gif_every_x_epochs == 0:
                self.visualize_and_save_game_state(episode, game_action, probs)

        return np.sum(rewards), score

    def visualize_and_save_game_state(self, episode, game_action, probs):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            image = self.visualizer.save_current_frame(game_action, probs)
            self.frames.append(image)

    def print_epoch_summary(self, episode, relevant_rewards, relevant_scores, validation=False):
        min_reward = int(np.nanmin(relevant_rewards))
        max_reward = int(np.nanmax(relevant_rewards))
        mean_reward = int(np.nanmean(relevant_rewards))
        mean_score = np.nanmean(relevant_scores)
        med_score = np.nanmedian(relevant_scores)
        max_score = np.nanmax(relevant_scores)
        if not validation:
            print(
                f"Episodes {episode + 1 - self.n_memory_episodes}-{episode + 1}: Min Reward: {min_reward},"
                f" Max Reward: {max_reward}, Mean Reward: {mean_reward}, Mean Score: {mean_score},"
                f" Median Score: {med_score}, Max Score: {max_score}")
        else:
            print(
                f"Episode {episode +1} Validation: Min Reward: {min_reward},"
                f" Max Reward: {max_reward}, Mean Reward: {mean_reward}, Mean Score: {mean_score},"
                f" Median Score: {med_score}, Max Score: {max_score}, N Validation Games: {len(relevant_scores)}")

    def log_and_compile_gif(self, episode):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            gif_filename = f"{self.prefix_name}episode_{episode + 1}_score_{self.score_memory[-1]}.gif"
            imageio.mimsave(gif_filename, self.frames, fps=5)
            print(f"GIF saved for episode {episode + 1}.")
            self.frames = []  # Clear frames after saving

        if (episode + 1) % self.n_memory_episodes == 0:
            relevant_rewards = self.rewards_memory[-self.n_memory_episodes:]
            relevant_scores = self.score_memory[-self.n_memory_episodes:]
            self.print_epoch_summary(episode, relevant_rewards, relevant_scores)
            self.rewards_memory = []

            if len(self.score_memory) >= 5 * self.n_memory_episodes and self.use_scheduler:
                recent_mean_score = np.nanmean(self.score_memory[-5*self.n_memory_episodes:])
                self.scheduler.step(recent_mean_score)
                for param_group in self.optimizer.param_groups:
                    print("Current LR:", param_group['lr'])

    def validate_score(self, episode):
        rewards, scores, done = [], [], False
        last_start_prob = self.game.default_start_prob
        self.game.default_start_prob = 1
        self.model.eval()
        for validation_episode in range(self.validate_episodes):
            self.init_game_max_food_distance(episode)
            self.game.reset_game()
            state, last_action, last_score, done = preprocess_state(self.game), self.game.snake_direction, 0, False
            score, total_reward, steps = 0, 0, np.nan

            with torch.no_grad():
                while not done:
                    steps += 1
                    action, probs = self.choose_action(state, validation_episode, True)
                    game_action = postprocess_action(action)
                    self.game.change_direction(game_action)
                    score, done = self.game.move()
                    if self.check_failed_init(steps, done, -1, game_action, probs, last_action):
                        total_reward = np.nan
                        break

                    reward = self.compute_reward(score, last_score, done, last_action != game_action,
                                                 len(self.game.snake))
                    last_score = score
                    total_reward += reward
                    if validation_episode == 0:
                        self.visualize_and_save_game_state(self.save_gif_every_x_epochs+1, game_action, probs)
            scores.append(score)
            rewards.append(total_reward)
            print(" " * 100, end="\r")
            print(f"current validation reward: {total_reward}, current score: {score},"
                  f" n validation games: {len(scores)}", end="\r")
        self.game.default_start_prob = last_start_prob
        self.print_epoch_summary(episode, rewards, scores, True)
        if self.increasing_start_len:
            self.max_init_len = max(np.nanmean(scores)+1, 2)
        self.model.train()

        gif_filename = f"{self.prefix_name}val_episode_{episode + 1}_score_{self.score_memory[-1]}.gif"
        imageio.mimsave(gif_filename, self.frames, fps=5)
        print(f"GIF saved for episode {episode + 1}.")

    def train(self):
        for episode in range(self.episodes):
            total_reward, score = self.run_episode(episode)
            print(" " * 100, end="\r")
            print(f"current reward: {total_reward}, current score: {score}", end="\r")
            self.rewards_memory.append(total_reward)
            self.score_memory.append(score)

            # Log performance statistics and compile GIF as configured
            self.log_and_compile_gif(episode)
            if (episode+1) % self.validate_every_n_episodes == 0:
                self.validate_score(episode)