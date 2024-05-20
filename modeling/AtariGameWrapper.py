import numpy as np


class AtariGameWrapper:
    def __init__(self, game):
        self.game = game
        self.episode_rewards = []

    def get_score(self):
        return np.sum(self.episode_rewards)

    def step(self):
        obs, reward, done, trunc, info = self.game.step()
        self.episode_rewards.append(reward)
        return obs, reward, done, trunc, info

    def reset(self):
        self.game.reset()
        self.episode_rewards = []

    def on_validation_end(self, episode, rewards, scores):
        pass
