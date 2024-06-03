import torch
import torch.nn.functional as F
import cv2
from collections import deque
import random
import numpy as np

class AtariGameViz:
    def __init__(self, game, device):
        self.game = game
        self.device = device

    def save_current_frame(self, game_action, probs):
        img = self.game.render()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.25
        thickness = 1
        img_width = img.shape[1]
        text_size_action = cv2.getTextSize(f"Action: {game_action}", font, font_scale, thickness)[0]
        text_size_probs = cv2.getTextSize(f"Probs: {probs}", font, font_scale, thickness)[0]
        extra_height = max(text_size_action[1], text_size_probs[1]) + 20
        new_img = np.ones((img.shape[0] + extra_height, img.shape[1], 3), dtype=np.uint8) * 255
        new_img[extra_height:, :] = img
        x_center_action = (img_width - text_size_action[0]) // 2
        x_center_probs = (img_width - text_size_probs[0]) // 2

        cv2.putText(new_img, f"Action: {game_action}", (x_center_action, 10), font, font_scale, (125, 125, 125),
                    thickness, cv2.LINE_AA)
        cv2.putText(new_img, f"Probs: {probs}", (x_center_probs, 20), font, font_scale, (125, 125, 125), thickness,
                    cv2.LINE_AA)

        return new_img

    @staticmethod
    def pad_frames_to_same_size(frames):
        return frames


class Preprocessor:
    def __init__(self, config, device):
        self.resize_img = config['resize_img']
        self.gray_scale = config['gray_scale']
        self.normalize_factor = float(config['normalize_factor'])
        self.device = device

    def preprocess_state(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
        elif len(obs.shape) == 3:
            obs = obs.permute(2, 0, 1)
        if self.resize_img:
            obs = F.interpolate(obs.unsqueeze(0), size=self.resize_img[::-1], mode='bicubic', align_corners=False).squeeze(0)
        if self.gray_scale and obs.shape[0] == 3:
            obs = obs.mean(dim=0, keepdim=True)
        obs = obs / self.normalize_factor
        return obs

    @staticmethod
    def postprocess_action(action):
        return action


class AtariGameWrapper:
    def __init__(self, game, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = game
        self.episode_rewards = []
        self.preprocessor = Preprocessor(config, self.device)
        self.env_memory = deque([], maxlen=int(config['random_envs']))
        self.default_start_prob = float(config['default_start_prob'])
        self.random_steps_range = tuple(config['random_steps_range'])
        self.stack_n_frames = int(config['stack_n_frames'])
        self.losing_live_penalty = int(config.get("losing_live_penalty", 0))
        self.initial_frame_skip = int(config.get("initial_frame_skip", 0))
        self.lives = 0
        self.stacked_frames = deque([], maxlen=self.stack_n_frames)

    def init_random_start(self):
        obs, info = self.game.reset()
        for _ in range(random.randint(*self.random_steps_range)):
            action = self.game.action_space.sample()
            obs, reward, done, trunc, info = self.game.step(action)
            if done:
                obs, info = self.game.reset()
        return obs, info

    def get_score(self):
        return sum(self.episode_rewards)

    def step(self, action):
        obs, reward, done, trunc, info = self.game.step(action)
        self.episode_rewards.append(reward)
        if self.lives > info.get('lives', 0):
            reward -= self.losing_live_penalty
            self.lives = info.get('lives', 0)
        obs = self.preprocessor.preprocess_state(obs)
        if self.stack_n_frames > 0:
            self.stacked_frames.append(obs)
            obs = torch.cat(list(self.stacked_frames), dim=0)
        return obs, reward, done, trunc, info

    def init_rand_pos(self):
        low = self.game.observation_space.low
        high = self.game.observation_space.high
        random_start_state = torch.tensor(np.random.uniform(low, high), dtype=torch.float32, device=self.device)
        state, info = self.game.reset()
        self.game.unwrapped.state = random_start_state.cpu().numpy().astype(state.dtype)
        obs = self.game.unwrapped.state
        obs = self.preprocessor.preprocess_state(obs)
        return obs, info

    def reset(self, options={}):
        self.episode_rewards = []
        rand_start = False
        validation = options.get('validation', False)
        if options.get('randomize_position', False) and not validation:
            obs, info = self.init_rand_pos()
            self.lives = info.get('lives')
            return obs, info
        if random.random() > self.default_start_prob and not validation:
            obs, info = self.init_random_start()
            rand_start = True
        else:
            obs, info = self.game.reset()
        obs = self.preprocessor.preprocess_state(obs)
        if self.stack_n_frames > 0:
            self.stacked_frames = deque([obs] * self.stack_n_frames, maxlen=self.stack_n_frames)
            obs = torch.cat(list(self.stacked_frames), dim=0)
        self.lives = info.get('lives', 0)
        if not rand_start:
            for i in range(self.initial_frame_skip):
                obs, _, _, _, info = self.step(0)
        return obs, info

    def on_validation_end(self, episode, rewards, scores):
        pass
