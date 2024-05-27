import numpy as np
import cv2
from collections import deque
import random

class AtariGameViz:
    def __init__(self, game):
        self.game = game

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
    def __init__(self, config):
        self.resize_img = config['resize_img']
        self.gray_scale = config['gray_scale']

    def preprocess_state(self, obs):
        if self.resize_img:
            obs = cv2.resize(obs, dsize=self.resize_img, interpolation=cv2.INTER_CUBIC)
        if self.gray_scale and len(obs.shape) == 3:
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
    def __init__(self, game, config):
        self.game = game
        self.episode_rewards = []
        self.preprocessor = Preprocessor(config)
        self.env_memory = deque([], maxlen=int(config['random_envs']))
        self.default_start_prob = float(config['default_start_prob'])
        self.random_steps_range = tuple(config['random_steps_range'])
        self.stack_n_frames = int(config['stack_n_frames'])
        self.stacked_frames = deque([], maxlen=self.stack_n_frames)

    def init_random_start(self):
        obs, info = self.game.reset(seed=np.random.randint(0, 10000))
        for _ in range(random.randint(*self.random_steps_range)):
            action = self.game.action_space.sample()
            obs, reward, done, trunc, info = self.game.step(action)
            if done:
                obs, info = self.game.reset(seed=np.random.randint(0, 10000))
        return obs, info

    def get_score(self):
        return np.sum(self.episode_rewards)

    def step(self, action):
        obs, reward, done, trunc, info = self.game.step(action)
        obs = self.preprocessor.preprocess_state(obs) # obs[5:13, 65:85] skiing flags left
        if self.stack_n_frames > 0:
            self.stacked_frames.append(obs)
            obs = np.vstack(self.stacked_frames)
        self.episode_rewards.append(reward)
        return obs, reward, done, trunc, info

    def init_rand_pos(self):
        low = self.game.observation_space.low
        high = self.game.observation_space.high
        random_start_state = np.random.uniform(low, high)
        state, info = self.game.reset(seed=np.random.randint(0, 10000))
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
            obs, info = self.game.reset(seed=np.random.randint(0, 10000))
        obs = self.preprocessor.preprocess_state(obs)
        if self.stack_n_frames > 0:
            self.stacked_frames = deque([obs]*self.stack_n_frames, maxlen=self.stack_n_frames)
            obs = np.vstack(self.stacked_frames)
        return obs, info

    def on_validation_end(self, episode, rewards, scores):
        pass
