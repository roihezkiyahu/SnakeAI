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
import csv
from trainer import Trainer

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
                 batch_size=1024,
                 EPS_START=1,
                 EPS_END=0,
                 EPS_DECAY=250,
                 max_episode_len=10000,
                 close_food=2500,
                 close_food_episodes_skip=100,
                 max_init_len=15,
                 replaymemory=10000,
                 discount_rate=0.99,
                 per_alpha=0.6,
                 validate_every_n_episodes=500,
                 validate_episodes=100,
                 increasing_start_len=False,
                 patience=3,
                 entropy_coefficient=0.01
                 ):
        super().__init__(game, value_network, actor_network, gamma=gamma, reward_params=reward_params,
                         max_init_len=max_init_len, close_food=close_food,
                         close_food_episodes_skip=close_food_episodes_skip, increasing_start_len=increasing_start_len,
                         n_memory_episodes=n_memory_episodes, prefix_name=prefix_name, folder=folder,
                         save_gif_every_x_epochs=save_gif_every_x_epochs, batch_size=batch_size,
                         max_episode_len=max_episode_len, discount_rate=discount_rate,
                         validate_every_n_episodes=validate_every_n_episodes, validate_episodes=validate_episodes)
        self.value_network = value_network.to(self.device)
        self.actor_network = actor_network.to(self.device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=value_network_lr)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_network_lr)
        self.entropy_coefficient = entropy_coefficient

    def _returns_advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self, epochs, batch_size):
        episode_count = 0
        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = np.empty((batch_size,) + (self.game.width, self.game.height), dtype=np.float)
        obs = self.init_episode()
        last_action = self.game.snake_direction
        total_reward = 0
        steps = 1

        for epoch in range(epochs):
            for i in range(batch_size):
                observations[i] = obs
                values[i] = self.value_network(torch.tensor(obs, dtype=torch.float).to(self.device)).detach().numpy()
                policy = self.actor_network(torch.tensor(obs, dtype=torch.float).to(self.device))
                actions[i] = torch.multinomial(policy, 1).detach().numpy()
                game_action = postprocess_action(actions[i])
                self.game.change_direction(game_action)
                score, done = self.game.move()
                if self.check_failed_init(steps, done, epoch, game_action, policy, last_action):
                    rewards[i] = np.nan
                    break
                reward = self.compute_reward(score, last_score, done, last_action != game_action, len(self.game.snake))
                last_score = score
                total_reward += reward
                obs, rewards[i], dones[i], _ = preprocess_state(self.game), reward, done

                if dones[i]:
                    obs = self.init_episode()

                if (episode_count + 1) % self.save_gif_every_x_epochs == 0:
                    self.visualize_and_save_game_state(episode_count, game_action, policy)
                steps +=1
                if steps >= self.max_episode_len:
                    done = True
                    obs = self.init_episode()
                if done:
                    print(" " * 100, end="\r")
                    print(f"current reward: {total_reward}, current score: {score}", end="\r")
                    self.rewards_memory.append(total_reward)
                    self.score_memory.append(score)
                    self.log_and_compile_gif(episode_count)
                    episode_count += 1
                    total_reward = 0
                    steps = 0

            # If our epiosde didn't end on the last step we need to compute the value for the last state
            if dones[-1]:
                next_value = 0
            else:
                next_value = self.value_network(torch.tensor(obs, dtype=torch.float)).detach().numpy()[0]

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # Learning step !
            self.optimize_model(observations, actions, returns, advantages)

            if (epoch+1) % self.validate_every_n_episodes == 0 or epoch == epochs - 1:
                self.model = self.actor_network()
                self.validate_score(episode_count)

        print(f'The trainnig was done over a total of {episode_count} episodes')

    def optimize_model(self, observations, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions), self.env.action_space.n).to(torch.float).to(self.device)
        returns = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        observations = torch.tensor(observations, dtype=torch.float).to(self.device)

        logits, values = self.model(observations)

        value_loss = F.mse_loss(values.squeeze(), returns.squeeze()).to(self.device)

        probs = F.softmax(logits, dim=-1)
        action_log_probs = torch.log(torch.sum(probs * actions, dim=1))
        actor_loss = -(action_log_probs * advantages).mean().to(self.device)

        entropy = -(probs * torch.log(probs)).sum(-1).mean().to(self.device)

        total_loss = value_loss + actor_loss - self.entropy_coefficient * entropy

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), actor_loss.item(), value_loss.item(), entropy.item()