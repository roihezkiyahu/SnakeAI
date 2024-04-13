import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import imageio
import numpy as np
try:
    import cv2
except:
    print("no opencv moudle")

class GameVisualizer:
    def __init__(self, game):
        self.game = game

    def draw_grid(self, ax):
        # Extend the grid to include the border
        for x in range(-1, self.game.width + 1):
            for y in range(-1, self.game.height + 1):
                if x == -1 or y == -1 or x == self.game.width or y == self.game.height:
                    # Draw the black border
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor='black')
                else:
                    # Draw the game grid
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='gray', facecolor='none')
                ax.add_patch(rect)

    def draw_snake(self, ax):
        # Offset snake positions by 1 to account for the new border
        for i, (x, y) in enumerate(self.game.snake):
            if i == 0:  # Head of the snake
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='darkgreen'))
            else:  # Body of the snake
                ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor='green'))

    def draw_food(self, ax):
        # Offset food position by 1
        food_x, food_y = self.game.food
        ax.add_patch(patches.Circle((food_x + 0.5, food_y + 0.5), 0.5, facecolor='red'))

    def save_current_frame(self, game_action, probs):
        fig, ax = plt.subplots(figsize=(5, 5))
        # Adjust the limits to include the new border
        ax.set_xlim(-1, self.game.width + 1)
        ax.set_ylim(-1, self.game.height + 1)
        plt.axis('off')

        self.draw_grid(ax)
        self.draw_snake(ax)
        self.draw_food(ax)

        buffer = BytesIO()
        if game_action in ["UP", "DOWN"]:
            game_action = {'UP': 'DOWN', 'DOWN': 'UP'}[game_action]
        plt.title(f"action: {game_action}, probs: {probs}")
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buffer.seek(0)
        image = imageio.imread(buffer)
        buffer.close()
        return image


    def pad_frames_to_same_size(self, frames):
        # Find the max dimensions
        max_height = max(frame.shape[0] for frame in frames)
        max_width = max(frame.shape[1] for frame in frames)

        # Pad the images
        padded_frames = []
        for frame in frames:
            pad_height = (max_height - frame.shape[0]) // 2
            pad_width = (max_width - frame.shape[1]) // 2

            padded_frame = np.pad(frame,
                                  ((pad_height, max_height - frame.shape[0] - pad_height),
                                   (pad_width, max_width - frame.shape[1] - pad_width),
                                   (0, 0)),
                                  mode='constant', constant_values=0)
            padded_frames.append(padded_frame)

        return padded_frames


class GameVisualizer_cv2:
    def __init__(self, game):
        self.game = game

    def draw_grid(self, img, offset_y):
        # Draw the black border around the grid
        cv2.rectangle(img, (0, offset_y), (img.shape[1], img.shape[0]), (0, 0, 0), thickness=1)
        # Draw the grid lines
        for x in range(1, self.game.width):
            cv2.line(img, (x * self.cell_size, offset_y), (x * self.cell_size, offset_y + self.game.height * self.cell_size), (200, 200, 200), 1)
        for y in range(1, self.game.height):
            cv2.line(img, (0, offset_y + y * self.cell_size), (self.game.width * self.cell_size, offset_y + y * self.cell_size), (200, 200, 200), 1)

    def draw_snake(self, img, offset_y):
        for i, (x, y) in enumerate(self.game.snake):
            color = (0, 128, 0) if i == 0 else (0, 255, 0)  # Dark green for head, light green for body
            cv2.rectangle(img, (x * self.cell_size, y * self.cell_size + offset_y),
                          ((x + 1) * self.cell_size, (y + 1) * self.cell_size + offset_y), color, -1)

    def draw_food(self, img, offset_y):
        food_x, food_y = self.game.food
        center = (int(food_x * self.cell_size + self.cell_size / 2), int(food_y * self.cell_size + self.cell_size / 2 + offset_y))
        cv2.circle(img, center, self.cell_size // 2, (0, 0, 255), -1)  # Red color

    def save_current_frame(self, game_action, probs):
        cell_size = 60  # Increase cell size to make the image three times larger
        self.cell_size = cell_size
        text_height = 80  # Allocate more space for text if needed
        img_height = self.game.height * cell_size + text_height
        img_width = self.game.width * cell_size
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img.fill(255)  # Set background to white

        self.draw_grid(img, text_height)
        self.draw_snake(img, text_height)
        self.draw_food(img, text_height)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_size_action = cv2.getTextSize(f"Action: {game_action}", font, font_scale, thickness)[0]
        text_size_probs = cv2.getTextSize(f"Probs: {probs}", font, font_scale, thickness)[0]
        x_center_action = (img_width - text_size_action[0]) // 2
        x_center_probs = (img_width - text_size_probs[0]) // 2

        cv2.putText(img, f"Action: {game_action}", (x_center_action, 30), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(img, f"Probs: {probs}", (x_center_probs, 60), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return img

    def pad_frames_to_same_size(self, frames):
        max_height = max(frame.shape[0] for frame in frames)
        max_width = max(frame.shape[1] for frame in frames)
        padded_frames = []
        for frame in frames:
            pad_height = (max_height - frame.shape[0]) // 2
            pad_width = (max_width - frame.shape[1]) // 2
            padded_frame = cv2.copyMakeBorder(frame, pad_height, max_height - frame.shape[0] - pad_height,
                                              pad_width, max_width - frame.shape[1] - pad_width,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
            padded_frames.append(padded_frame)
        return padded_frames