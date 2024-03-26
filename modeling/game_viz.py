import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import imageio

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
