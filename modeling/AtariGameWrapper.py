import numpy as np


class AtariGameViz:
    def __init__(self, game):
        self.game = game

    def save_current_frame(self, game_action, probs):
        return self.game.render()

    @staticmethod
    def pad_frames_to_same_size(frames):
        return frames


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def postprocess_action(action):
        return action


class AtariGameWrapper:
    def __init__(self, game):
        self.game = game
        self.episode_rewards = []
        self.preprocessor = Preprocessor()

    def get_score(self):
        return np.sum(self.episode_rewards)

    def step(self, action):
        obs, reward, done, trunc, info = self.game.step(action)
        self.episode_rewards.append(reward)
        if len(obs.shape) >= 3:
            return np.moveaxis(obs, -1, 0), reward, done, trunc, info
        return obs, reward, done, trunc, info

    def reset(self, options=None):
        self.episode_rewards = []
        if options and options.get('randomize_position', False):
            low = self.game.observation_space.low
            high = self.game.observation_space.high
            random_start_state = np.random.uniform(low, high)
            state, info = self.game.reset()
            self.game.unwrapped.state = np.array(random_start_state, dtype=state.dtype)
            if len(self.game.unwrapped.state.shape) >= 3:
                return np.moveaxis(self.game.unwrapped.state, -1, 0), info
            return self.game.unwrapped.state, info
        state, info = self.game.reset()
        if len(state.shape) >= 3:
            return np.moveaxis(state, -1, 0), info
        return state, info

    def on_validation_end(self, episode, rewards, scores):
        pass

