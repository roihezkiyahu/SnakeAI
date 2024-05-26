import numpy as np
import cv2

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

