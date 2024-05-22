from modeling.game_viz import GameVisualizer, GameVisualizer_cv2
import torch
import torch.nn as nn
import torch.optim as optim
import os
from modeling.memory import ReplayMemory, Transition, PERMemory
import math
import numpy as np
from modeling.processing import preprocess_state, postprocess_action
import torch.nn.functional as F
import random
import imageio
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print("no TkAgg")
import matplotlib.pyplot as plt
import csv
from modeling.AtariGameWrapper import AtariGameWrapper, AtariGameViz

class Trainer:
    def __init__(
            self,
            game,
            model,
            clone_model,
            episodes=10000,
            learning_rate=5e-5,
            gamma=0.99,
            n_memory_episodes=100,
            prefix_name="",
            folder="",
            save_gif_every_x_epochs=500,
            batch_size=512,
            EPS_START=1,
            EPS_END=0,
            EPS_DECAY=250,
            TAU=5e-3,
            max_episode_len=10000,
            use_ddqn=True,
            replaymemory=10000,
            optimizer=None,
            discount_rate=0.99,
            per_alpha=0.6,
            use_scheduler=False,
            validate_every_n_episodes=500,
            validate_episodes=100,
            patience=3,
            n_actions=4,
            game_wrapper=None,
            visualizer=None,
            gif_fps=5,
            reset_options=None,
            update_every_n_steps=1
    ):
        if isinstance(game_wrapper, type(None)):
            game_wrapper = AtariGameWrapper(game)
        if isinstance(visualizer, type(None)):
            visualizer = AtariGameViz(game)
        self.use_ddqn = use_ddqn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = game
        self.visualizer = visualizer
        self.model = model.to(self.device)
        self.target_net = clone_model.to(self.device)
        self.episodes = episodes
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_memory_episodes = n_memory_episodes
        self.batch_size = batch_size
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.max_episode_len = max_episode_len
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
                                                                        patience=patience, threshold=0.1, verbose=True)

        self.criterion = nn.MSELoss()
        self.rewards_memory = []
        self.score_memory = []
        self.frames = []
        self.memory = PERMemory(replaymemory, per_alpha)
        self.discount_rate = discount_rate
        self.validate_every_n_episodes = validate_every_n_episodes
        self.validate_episodes = validate_episodes
        self.validation_log = []
        self.n_actions = n_actions
        self.game_wrapper = game_wrapper
        self.fps = gif_fps
        self.reset_options = reset_options
        self.update_every_n_steps = update_every_n_steps

    def choose_action(self, state, episode, validation=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if validation:
            with torch.no_grad():
                probs = self.model(state_tensor)
                return torch.argmax(probs).item(), torch.round(F.softmax(probs[0]) * 100).cpu().int().tolist()
        else:
            epsilon = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * episode / self.EPS_DECAY)
            if torch.rand(1, device=self.device).item() < epsilon:
                return torch.randint(0, self.n_actions, (1,), device=self.device).item(), [0]*self.n_actions
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

    def calc_loss(self):
        state_batch, action_batch, reward_batch, next_state_batch, non_final_mask, weights, indices = self.get_batch()
        state_action_values = self.model(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = (weights * F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1),
                                           reduction='none')).mean()
        return loss, expected_state_action_values, state_action_values, indices

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        loss, expected_state_action_values, state_action_values, indices = self.calc_loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            td_errors = (expected_state_action_values.unsqueeze(1) - state_action_values).abs().squeeze().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

    def check_failed_init(self, steps, done, episode, action, probs, debug=False):
        game_action = self.game_wrapper.preprocessor.postprocess_action(action)
        if steps <= 1 and done:
            if debug:
                print("failed attempt")
                image = self.visualizer.save_current_frame(game_action, probs)
                plt.imshow(image)
                plt.show()
            if (episode + 1) % self.save_gif_every_x_epochs == 0:
                self.visualize_and_save_game_state(episode, game_action, probs)
            return True
        return False

    @staticmethod
    def get_next_state_tensor(terminated, obs):
        if terminated:
            return None
        return torch.FloatTensor(obs).unsqueeze(0)

    def update_target_net(self):
        target_dict, policy_dict = self.target_net.state_dict(), self.model.state_dict()
        for key in policy_dict:
            target_dict[key] = policy_dict[key] * self.TAU + target_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_dict)

    def save_gif_if_needed(self, episode, action, probs):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            self.visualize_and_save_game_state(
                episode,
                self.game_wrapper.preprocessor.postprocess_action(action),
                probs
            )

    def process_episode_step(self, steps, episode, action, probs, reward, terminated, truncated, state, obs):
        done = terminated or truncated
        if self.check_failed_init(steps, done, episode, action, probs):
            return done, True
        next_state_tensor = self.get_next_state_tensor(terminated, obs)
        self.memory.push(torch.tensor(state, device=self.device).to(torch.float32), torch.tensor([action], device=self.device),
                         next_state_tensor, torch.tensor([reward], device=self.device))
        if (steps + 1) % self.update_every_n_steps == 0:
            self.optimize_model()
            self.update_target_net()
        self.save_gif_if_needed(episode, action, probs)
        return done, False

    def run_episode(self, episode):
        state, info = self.game_wrapper.reset(self.reset_options)
        done, steps, rewards = False, 0, []
        while not done and steps <= self.max_episode_len:
            steps += 1
            action, probs = self.choose_action(state, episode)
            obs, reward, terminated, truncated, _ = self.game_wrapper.step(action)
            done, failed = self.process_episode_step(steps, episode, action, probs, reward,
                                                     terminated, truncated, state, obs)
            if failed:
                break
            rewards.append(reward)
            state = obs
        return np.sum(rewards), self.game_wrapper.get_score()

    def visualize_and_save_game_state(self, episode, game_action, probs):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            image = self.visualizer.save_current_frame(game_action, probs)
            self.frames.append(image)

    def print_epoch_summary(self, episode, relevant_rewards, relevant_scores, validation=False):
        min_reward, max_reward = int(np.nanmin(relevant_rewards)), int(np.nanmax(relevant_rewards))
        mean_reward, mean_score = int(np.nanmean(relevant_rewards)), np.nanmean(relevant_scores)
        med_score, max_score = np.nanmedian(relevant_scores), np.nanmax(relevant_scores)
        if not validation:
            print(
                f"Episodes {episode + 1 - self.n_memory_episodes}-{episode + 1}: Min Reward: {min_reward:.2f},"
                f" Max Reward: {max_reward:.2f}, Mean Reward: {mean_reward:.2f}, Mean Score: {mean_score:.2f},"
                f" Median Score: {med_score:.2f}, Max Score: {max_score:.2f}")
        else:
            print(
                f"Episode {episode +1} Validation: Min Reward: {min_reward:.2f},"
                f" Max Reward: {max_reward:.2f}, Mean Reward: {mean_reward:.2f}, Mean Score: {mean_score:.2f},"
                f" Median Score: {med_score:.2f}, Max Score: {max_score:.2f}, N Validation Games: {len(relevant_scores)}")

    def log_and_compile_gif(self, episode):
        if (episode + 1) % self.save_gif_every_x_epochs == 0:
            gif_filename = f"{self.prefix_name}episode_{episode + 1}_score_{self.score_memory[-1]:.2f}.gif"
            imageio.mimsave(gif_filename, self.visualizer.pad_frames_to_same_size(self.frames), fps=self.fps, loop=0)
            print(f"GIF saved for episode {episode + 1}.")
            self.frames = []  # Clear frames after saving

        if (episode + 1) % self.n_memory_episodes == 0:
            relevant_rewards = self.rewards_memory[-self.n_memory_episodes:]
            relevant_scores = self.score_memory[-self.n_memory_episodes:]
            self.print_epoch_summary(episode, relevant_rewards, relevant_scores)
            self.rewards_memory = []

    def run_validation_move(self, steps, state, validation_episode, total_reward,
                            viz_total_reward, viz_score):
        action, probs = self.choose_action(state, validation_episode, True)
        obs, reward, terminated, truncated, _ = self.game_wrapper.step(action)
        done = terminated or truncated
        if self.check_failed_init(steps, done, -10 if validation_episode != 0 else -1, action, probs):
            return True, np.nan, self.game_wrapper.get_score(), obs, viz_total_reward, viz_score
        total_reward += reward
        if validation_episode == 0:
            self.visualize_and_save_game_state(self.save_gif_every_x_epochs - 1,
                                               self.game_wrapper.preprocessor.postprocess_action(action), probs)
            viz_total_reward, viz_score = total_reward, self.game_wrapper.get_score()
        return done, total_reward, self.game_wrapper.get_score(), obs, viz_total_reward, viz_score

    @staticmethod
    def pp_val_episode(scores, score, rewards, total_reward, validation_episode):
        scores.append(score)
        rewards.append(total_reward)
        print(" " * 100, end="\r")
        print(f"val episode: {validation_episode + 1}, current validation reward: {total_reward:.2f},"
              f" current score: {score:.2f}, n validation games: {len(scores)}", end="\r")
        return scores, rewards

    def save_validation_gif(self, episode, viz_score):
        gif_filename = f"{self.prefix_name}val_episode_{episode + 1}_score_{viz_score}.gif"
        imageio.mimsave(gif_filename, self.visualizer.pad_frames_to_same_size(self.frames), fps=self.fps, loop=0)
        print(f"GIF saved for episode {episode + 1}.")
        self.frames = []

    def validate_score(self, episode):
        rewards, scores, done, viz_total_reward, viz_score = [], [], False, 0, 0
        for validation_episode in range(self.validate_episodes):
            self.model.eval()
            state, info = self.game_wrapper.reset(options={"validation": True})
            score, total_reward, steps, done = 0, 0, 0, False
            with torch.no_grad():
                while not done and steps <= self.max_episode_len:
                    steps += 1
                    run_res = self.run_validation_move(steps, state, validation_episode, total_reward,
                                                       viz_total_reward, viz_score)
                    done, total_reward, score, state, viz_total_reward, viz_score = run_res
                scores, rewards = self.pp_val_episode(scores, score, rewards, total_reward, validation_episode)
        self.on_validation_end(episode, rewards, scores, viz_score)

    def save_validation_csv(self):
        csv_filename = f"{self.prefix_name}validation_summary.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['episode', 'Mean Score', 'Median Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for log_entry in self.validation_log:
                writer.writerow(log_entry)
        print(f"Validation summary saved to {csv_filename}")

    def plot_validation_convergence(self):
        episodes = [log_entry['episode'] for log_entry in self.validation_log]
        mean_scores = [log_entry['Mean Score'] for log_entry in self.validation_log]
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, mean_scores, marker='o', linestyle='-', color='b')
        plt.title('Validation Score Convergence')
        plt.xlabel('Episode')
        plt.ylabel('Mean Score')
        plt.grid(True)
        plot_filename = f"{self.prefix_name}validation_convergence.png"
        plt.savefig(plot_filename)
        plt.close()  # Close the figure to free memory
        print(f"Convergence plot saved to {plot_filename}")

    def on_validation_end(self, episode, rewards, scores, viz_score):
        self.print_epoch_summary(episode, rewards, scores, True)
        mean_score = np.nanmean(scores)
        self.game_wrapper.on_validation_end(episode, rewards, scores)
        self.model.train()
        self.save_validation_gif(episode, viz_score)
        if self.use_scheduler:
            self.scheduler.step(np.nanmean(scores))
            for param_group in self.optimizer.param_groups:
                print("Current LR:", param_group['lr'])
        self.validation_log.append({"episode": episode+1, "Mean Score": mean_score, "Median Score": np.nanmedian(scores)})
        self.save_validation_csv()
        self.plot_validation_convergence()

    def train(self):
        for episode in range(self.episodes):
            total_reward, score = self.run_episode(episode)
            print(" " * 100, end="\r")
            print(f"current episode: {episode}, current reward: {total_reward}, current score: {score}", end="\r")
            if np.isnan(total_reward):
                print("debug")
            self.rewards_memory.append(total_reward)
            self.score_memory.append(score)

            # Log performance statistics and compile GIF as configured
            self.log_and_compile_gif(episode)
            if (episode+1) % self.validate_every_n_episodes == 0:
                self.validate_score(episode)