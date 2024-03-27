from game_viz import GameVisualizer
import torch
import torch.nn as nn
import torch.optim as optim
import os
from memory import ReplayMemory, Transition
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
            learning_rate=0.001,
            gamma=0.99,
            reward_params={'death': -25, 'move': 1, 'food': 100},
            n_memory_episodes=500,
            prefix_name="",
            folder="",
            save_gif_every_x_epochs=1000,
            batch_size=1024,
            EPS_START=0.95,
            EPS_END=0.05,
            EPS_DECAY=1000,
            TAU=0.005,
            max_episode_len=5000,
            close_food=1000,
            close_food_episodes_skip=100,
            use_ddqn=False,
            max_init_len=15,
            replaymemory=10000,
            optimizer=None,
            discount_rate=0.99
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
        self.criterion = nn.MSELoss()
        self.rewards_memory = []
        self.score_memory = []
        self.frames = []
        self.memory = ReplayMemory(replaymemory)
        self.max_init_len = max_init_len
        self.discount_rate = discount_rate

    def choose_action(self, state, episode):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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

    def compute_reward(self, score, last_score, done, move):
        reward = self.reward_params['death'] * done
        reward += self.reward_params['move'] * move
        reward += (score - last_score) * self.reward_params['food'] * (1 + (score - 1) / 2)
        return reward

    def update_state(self, done):
        if done:
            return None, None
        next_state = preprocess_state(self.game)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        return next_state, next_state_tensor

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool,
                                      device=self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Get the current Q-values for all actions
        state_action_values = self.model(state_batch).gather(1, action_batch)

        if self.use_ddqn:
            # In DDQN, select actions using the main network
            best_actions = self.model(non_final_next_states).max(1)[1].unsqueeze(1)  # [batch_size, 1] for gather
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            # Evaluate the selected actions' Q-values using the target network
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze()
        else:
            # In DQN, directly estimate the Q-values of next states using the target network
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Calculate the expected Q-values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def init_game_max_food_distance(self, episode):
        if episode > self.close_food:
            self.game.max_food_distance = None
        else:
            self.game.max_food_distance = episode // self.close_food_episodes_skip + 1

    def check_failed_init(self, steps, done, episode, game_action, probs, last_action, debug=False):
        if steps == 1 and done:
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
            if self.check_failed_init(self, steps, done, episode, game_action, probs, last_action):
                rewards.append(np.nan)
                break

            reward = self.compute_reward(score, last_score, done, last_action != game_action)
            last_score = score
            rewards.append(reward)

            next_state, next_state_tensor = self.update_state(done)
            self.memory.push(torch.tensor(state, device=self.device),
                             torch.tensor([action], device=self.device),
                             next_state_tensor, torch.tensor([reward], device=self.device))
            state = next_state
            self.optimize_model()

            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.model.state_dict()
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

    def log_and_compile_gif(self, episode):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            gif_filename = f"{self.prefix_name}episode_{episode + 1}_score_{self.rewards_memory[-1]}.gif"
            imageio.mimsave(gif_filename, self.frames, fps=5)
            print(f"GIF saved for episode {episode + 1}.")
            self.frames = []  # Clear frames after saving

        if (episode + 1) % self.n_memory_episodes == 0:
            relevant_rewards = self.rewards_memory[-self.n_memory_episodes:]
            min_reward = np.nanmin(relevant_rewards)
            max_reward = np.nanmax(relevant_rewards)
            mean_reward = np.nanmean(relevant_rewards)
            mean_score = np.nanmean(self.score_memory[-self.n_memory_episodes:])
            print(
                f"Episodes {episode + 1 - self.n_memory_episodes}-{episode + 1}: Min Reward: {min_reward},"
                f" Max Reward: {max_reward}, Mean Reward: {mean_reward}, Mean Score: {mean_score}")
            self.rewards_memory = []

    def train(self):
        for episode in range(self.episodes):
            total_reward, score = self.run_episode(episode)
            print(f"current reward: {total_reward}, current score: {score}", end="\r")
            self.rewards_memory.append(total_reward)
            self.score_memory.append(score)

            # Log performance statistics and compile GIF as configured
            self.log_and_compile_gif(episode)