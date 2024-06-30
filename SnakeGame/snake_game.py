import pygame
import random
from types import SimpleNamespace

class SnakeGame:
    """A simple snake game using Pygame."""

    def __init__(self, width=10, height=10, block_size=10, default_start_prob=0.25, max_food_distance=None):
        """
        Initialize the game window, snake, and food.

        Args:
            width (int): The width of the game grid.
            height (int): The height of the game grid.
            block_size (int): The size of each block in pixels.
            default_start_prob (float): Probability of starting with the default game state.
        """
        self.action_space = SimpleNamespace(**{"n": 4})
        self.width = width
        self.height = height
        self.block_size = block_size
        self.default_start_prob = default_start_prob
        self.max_food_distance = max_food_distance
        self.reset_game()
        self.snake_direction = "UP"
        self.score = 0
        self.food = None
        self.game_over = False
        self.snake = []

    def reset_game(self, initial_length=2):
        """
        Reset the game by initializing the game state.

        Args:
            initial_length (int, optional): The initial length of the snake. Defaults to 3.
        """
        # Decide between default start and random start based on the probability
        if random.random() < self.default_start_prob:
            self.default_start()
        else:
            self.random_start(initial_length)
        self.score = 0
        self.food = None
        self.place_food()
        self.game_over = False
        return self.get_game_state()

    def default_start(self):
        """Initialize the game with a default start position and direction for the snake."""
        self.snake = [(self.width // 2, self.height // 2)]
        self.snake.append((self.width // 2, self.height // 2+1))
        self.snake_direction = 'UP'

    def random_start(self, initial_length):
        """Initialize the game with a random start position, direction, and length for the snake."""
        head_x = random.randint(2, self.width - 3)
        head_y = random.randint(2, self.height - 3)
        self.snake = [(head_x, head_y)]
        self.snake_direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])

        # Adjust initial length to ensure it's at least 1
        initial_length = max(2, initial_length)

        # Generate the rest of the snake body avoiding immediate illegal moves
        directions = {
            'UP': [(0, 1), (-1, 0), (1, 0)],  # Can't go DOWN
            'DOWN': [(0, -1), (-1, 0), (1, 0)],  # Can't go UP
            'LEFT': [(1, 0), (0, -1), (0, 1)],  # Can't go RIGHT
            'RIGHT': [(-1, 0), (0, -1), (0, 1)]  # Can't go LEFT
        }[self.snake_direction]

        for _ in range(1, initial_length):
            tail_x, tail_y = self.snake[-1]
            legal_positions = []

            # Check for legal positions around the current tail
            for dx, dy in directions:
                new_x, new_y = tail_x + dx, tail_y + dy
                if (new_x, new_y) not in self.snake and 0 < new_x < self.width - 1 and 0 < new_y < self.height - 1:
                    legal_positions.append((new_x, new_y))

            # If there are no legal positions, stop extending the body
            if not legal_positions:
                break

            # Choose a legal position for the next body segment
            self.snake.append(random.choice(legal_positions))

    def place_food(self):
        if self.max_food_distance is None:
            all_positions = {(x, y) for x in range(self.width) for y in range(self.height)}
            available_positions = list(all_positions - set(self.snake))
            if available_positions:
                self.food = random.choice(available_positions)
            else:
                self.food = None
            return

        head_x, head_y = self.snake[0]
        min_x, max_x = max(0, head_x - self.max_food_distance), min(self.width, head_x + self.max_food_distance + 1)
        min_y, max_y = max(0, head_y - self.max_food_distance), min(self.height, head_y + self.max_food_distance + 1)

        possible_positions = [(x, y) for x in range(min_x, max_x) for y in range(min_y, max_y)
                              if (x, y) not in self.snake and
                              ((x - head_x) ** 2 + (y - head_y) ** 2) <= self.max_food_distance ** 2]

        self.food = random.choice(possible_positions) if possible_positions else (
            random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def move(self):
        """
        Move the snake in the current direction. Update the game state, including checking for game over conditions.

        Returns:
            tuple: A tuple containing the current score and a boolean indicating if the game is over.
        """
        if self.game_over:
            return self.score, self.game_over

        head_x, head_y = self.snake[0]
        if self.snake_direction == 'UP':
            new_head = (head_x, head_y - 1)
        elif self.snake_direction == 'DOWN':
            new_head = (head_x, head_y + 1)
        elif self.snake_direction == 'LEFT':
            new_head = (head_x - 1, head_y)
        elif self.snake_direction == 'RIGHT':
            new_head = (head_x + 1, head_y)

        if new_head[0] < 0 or new_head[0] >= self.width or new_head[1] < 0 or new_head[
                1] >= self.height or new_head in self.snake:
            self.game_over = True
            return self.score, self.game_over

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

        if isinstance(self.food, type(None)): #won the game
            self.score += 10
            return self.score, True

        return self.score, self.game_over

    def change_direction(self, direction):
        """
        Change the direction of the snake's movement.

        Args:
            direction (str): The new direction for the snake to move in.
        """
        opposite_directions = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if direction in opposite_directions and not direction == opposite_directions[self.snake_direction]:
            self.snake_direction = direction

    def get_game_state(self):
        """
        Get the current state of the game.

        Returns:
            dict: A dictionary containing the snake's position, the food's position, and the current score.
        """
        return {
            'snake': self.snake,
            'food': self.food,
            'score': self.score
        }

    def close(self):
        pass


def draw_game(screen, game):
    """
    Draw the current game state to the Pygame window.

    Args:
        screen (pygame.Surface): The Pygame surface to draw the game state on.
        game (SnakeGame): The current game instance.
    """
    screen.fill((0, 0, 0))
    for pos in game.snake:
        rect = pygame.Rect(pos[0] * game.block_size, pos[1] * game.block_size, game.block_size, game.block_size)
        pygame.draw.rect(screen, (0, 255, 0), rect)
    food_rect = pygame.Rect(game.food[0] * game.block_size, game.food[1] * game.block_size, game.block_size,
                            game.block_size)
    pygame.draw.rect(screen, (255, 0, 0), food_rect)
    pygame.display.update()


def play_game():
    """
    Main function to start and run the game loop.
    """
    pygame.init()
    game = SnakeGame(20, 20, 20)
    screen = pygame.display.set_mode((game.width * game.block_size, game.height * game.block_size))
    clock = pygame.time.Clock()

    while not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.change_direction('UP')
                elif event.key == pygame.K_DOWN:
                    game.change_direction('DOWN')
                elif event.key == pygame.K_LEFT:
                    game.change_direction('LEFT')
                elif event.key == pygame.K_RIGHT:
                    game.change_direction('RIGHT')

        game.move()
        draw_game(screen, game)
        clock.tick(10)
    clock.tick(1)
    pygame.quit()
    print(f"Game over! Your score was: {game.score}")


if __name__ == "__main__":
    play_game()
