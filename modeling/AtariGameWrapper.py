import numpy as np
import cv2
from collections import deque
import random

class AtariGameViz:
    def __init__(self, game):
        self.game = game

    def save_current_frame(self, game_action, probs):
        return self.game.render()

    @staticmethod
    def pad_frames_to_same_size(frames):
        return frames


class Preprocessor:
    def __init__(self, resize_img=None, gray_scale=True):
        self.resize_img = resize_img
        self.gray_scale = gray_scale
        pass

    def preprocess_state(self, obs):
        if self.resize_img:
            obs = cv2.resize(obs, dsize=self.resize_img, interpolation=cv2.INTER_CUBIC)
        if self.gray_scale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if len(obs.shape) >= 3:
            return np.moveaxis(obs, -1, 0)
        if len(obs.shape) == 2:
            return np.expand_dims(obs, 0)
        return obs


    @staticmethod
    def postprocess_action(action):
        return action


class AtariGameWrapper:
    def __init__(self, game, resize_img=None, gray_scale=True, random_envs=0, default_start_prob=1,
                 random_steps_range=(200, 300), stack_n_frames=0):
        self.game = game
        self.episode_rewards = []
        self.preprocessor = Preprocessor(resize_img, gray_scale)
        self.resize_img = resize_img
        self.env_memory = deque([], maxlen=random_envs)
        self.default_start_prob = default_start_prob
        self.random_steps_range = random_steps_range
        self.stack_n_frames = stack_n_frames
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
        return np.sum(self.episode_rewards)

    def step(self, action):
        obs, reward, done, trunc, info = self.game.step(action)
        obs = self.preprocessor.preprocess_state(obs)
        if self.stack_n_frames > 0:
            self.stacked_frames.append(obs)
            obs = np.vstack(self.stacked_frames)
        self.episode_rewards.append(reward)
        return obs, reward, done, trunc, info

    def init_rand_pos(self):
        low = self.game.observation_space.low
        high = self.game.observation_space.high
        random_start_state = np.random.uniform(low, high)
        state, info = self.game.reset()
        self.game.unwrapped.state = np.array(random_start_state, dtype=state.dtype)
        obs = self.game.unwrapped.state
        self.preprocessor.preprocess_state(obs)
        return obs, info

    def reset(self, options={}):
        self.episode_rewards = []
        validation = options.get('validation', False)
        if options.get('randomize_position', False):
            return self.init_rand_pos()
        if random.random() > self.default_start_prob and not validation:
            obs, info = self.init_random_start()
        else:
            obs, info = self.game.reset()
        obs = self.preprocessor.preprocess_state(obs)
        if self.stack_n_frames > 0:
            self.stacked_frames = deque([obs]*self.stack_n_frames, maxlen=self.stack_n_frames)
            obs = np.vstack(self.stacked_frames)
        return obs, info

    def on_validation_end(self, episode, rewards, scores):
        pass

