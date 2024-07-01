import numpy as np
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque


class Preprocessor:
    def __init__(self, game, config):
        """
        Initialize the Preprocessor class.

        Args:
            game: An instance of the snake game.
            numeric_value (bool): Whether to return numeric values for food direction.
            for_cnn (bool): Whether the state is for a CNN.
            food_direction (bool): Whether to include food direction.
            add_death_indicators (bool): Whether to include death indicators.
            direction (bool): Whether to include direction indicators.
            clear_path_pixels (bool): Whether to include clear path pixels.
            length_aware (bool): Whether to include snake length.
        """
        self.game = game
        self.numeric_value = config['numeric_value']
        self.for_cnn = config['for_cnn']
        self.food_direction = config['food_direction']
        self.add_death_indicators = config['add_death_indicators']
        self.direction = config['direction']
        self.clear_path_pixels = config['clear_path_pixels']
        self.length_aware = config['length_aware']

    def initialize_state(self):
        """Initialize the state grid.

        Returns:
            np.array: A zero-initialized state grid.
        """
        state = np.zeros((self.game.height, self.game.width), dtype=np.float32)
        return state

    def mark_snake_on_state(self, state):
        """Mark the snake's body, head, and tail on the state grid.

        Args:
            state (np.array): The current state grid.

        Returns:
            np.array: The updated state grid with the snake marked.
        """
        for x, y in self.game.snake[1:-1]:
            state[y, x] = 11
        tail_x, tail_y = self.game.snake[-1]
        state[tail_y, tail_x] = 12
        head_x, head_y = self.game.snake[0]
        state[head_y, head_x] = 10
        return state

    def mark_food_on_state(self, state):
        """Mark the food's position on the state grid.

        Args:
            state (np.array): The current state grid.

        Returns:
            np.array: The updated state grid with the food marked.
        """
        if isinstance(self.game.food, type(None)): #won the game
            return state
        food_x, food_y = self.game.food
        state[food_y, food_x] = 5
        return state

    def make_border(self, state):
        """Add a border around the state grid.

        Args:
            state (np.array): The current state grid.

        Returns:
            np.array: The updated state grid with a border.
        """
        new_state = np.full((self.game.height + 2, self.game.width + 2), 20, dtype=np.float32)
        new_state[1:-1, 1:-1] = state
        return new_state

    def calculate_food_direction(self):
        """Calculate the direction and distance to the food from the snake's head.

        Returns:
            np.array: The direction and distance to the food.
        """
        head_x, head_y = self.game.snake[0]
        if isinstance(self.game.food, type(None)):  # won the game
            food_x, food_y = 0, 0
        else:
            food_x, food_y = self.game.food
        if self.numeric_value:
            left = food_x - head_x
            up = food_y - head_y
            radius = math.sqrt((food_x - head_x) ** 2 + (food_y - head_y) ** 2)
            theta = math.atan2(food_y - head_y, food_x - head_x)
            return np.array([left, up, radius, theta], dtype=np.float32)
        left = food_x < head_x
        right = food_x > head_x
        up = food_y > head_y
        down = food_y < head_y
        return np.array([left, right, up, down], dtype=np.float32)

    def calculate_clear_path(self, state):
        """Calculate the number of valid moves left in each direction until a collision.

        Args:
            state (np.array): The current state grid with border.

        Returns:
            np.array: The number of valid moves in each direction.
        """
        head_x, head_y = self.game.snake[0]
        no_border_game = state[1: -1, 1: -1]

        left = no_border_game[head_y, : head_x] if head_x != 0 else [0]
        right = no_border_game[head_y, head_x + 1:] if head_x != (self.game.width - 1) else [self.game.width - 1]
        free_left = np.isin(left, [0, 1], invert=True)
        free_right = np.isin(right, [0, 1], invert=True)
        free_left_moves = np.min(np.where(free_left[::-1])) if np.sum(free_left) > 0 else head_x
        free_right_moves = np.min(np.where(free_right)) if np.sum(free_right) > 0 else self.game.width - 1 - head_x

        up = no_border_game[:head_y, head_x] if head_y != 0 else [0]
        down = no_border_game[head_y + 1:, head_x] if head_y != (self.game.height - 1) else [self.game.height - 1]
        free_up = np.isin(up, [0, 1], invert=True)
        free_down = np.isin(down, [0, 1], invert=True)
        free_up_moves = np.min(np.where(free_up[::-1])) if np.sum(free_up) > 0 else head_y
        free_down_moves = np.min(np.where(free_down)) if np.sum(free_down) > 0 else self.game.height - 1 - head_y
        return np.array([free_left_moves, free_right_moves, free_up_moves, free_down_moves], dtype=np.float32)

    def calculate_death_indicators(self):
        """Calculate indicators for whether moving left, right, up, or down would result in death.

        Returns:
            np.array: The death indicators for each direction.
        """
        head_x, head_y = self.game.snake[0]
        death_indicators = np.zeros(4, dtype=np.float32)  # [left, right, up, down]
        for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            nx, ny = head_x + dx, head_y + dy
            if nx < 0 or nx >= self.game.width or ny < 0 or ny >= self.game.height or (nx, ny) in self.game.snake:
                death_indicators[i] = 1
        return death_indicators

    def calculate_direction_indicators(self):
        """Calculate the current direction of the snake.

        Returns:
            np.array: The current direction indicator.
        """
        direction = self.game.snake_direction
        return np.array([np.argmax(direction == ['UP', 'DOWN', 'LEFT', 'RIGHT'])], dtype=np.float32)

    def preprocess_state(self):
        """Convert the game state into a 2D grid suitable for CNN input, with optional features.

        Returns:
            np.array: The preprocessed state.
        """
        state = self.initialize_state()
        state = self.mark_snake_on_state(state)
        state = self.mark_food_on_state(state)
        state = self.make_border(state)

        additional_features = []

        if self.food_direction:
            additional_features.extend(self.calculate_food_direction())

        if self.length_aware:
            additional_features.extend(np.array([len(self.game.snake)], dtype=np.float32))

        if self.add_death_indicators:
            additional_features.extend(self.calculate_death_indicators())

        if self.direction:
            additional_features.extend(self.calculate_direction_indicators())

        if self.clear_path_pixels:
            additional_features.extend(self.calculate_clear_path(state))

        if self.for_cnn:
            state = state.reshape(self.game.height + 2, self.game.width + 2)
            if additional_features:
                additional_features_chanels = np.stack(
                    [np.full((self.game.height + 2, self.game.width + 2), feat) for feat in additional_features] + [state], axis=0)
                return additional_features_chanels
            return np.expand_dims(state, 0)
        else:
            state = state.flatten()
            if additional_features:
                state = np.concatenate([state, additional_features])

        return state.astype(np.float32)

    @staticmethod
    def postprocess_action(action):
        """Convert the neural network's output action into game action.

        Args:
            action (int): The action predicted by the neural network.

        Returns:
            str: The game action.
        """
        actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        return actions[action]


class SnakeGameWrap:
    def __init__(self, game, config):
        """
        Initialize the SnakeGameWrap class.

        Args:
            game: An instance of the snake game.
            numeric_value (bool): Whether to return numeric values for food direction.
            for_cnn (bool): Whether the state is for a CNN.
            food_direction (bool): Whether to include food direction.
            add_death_indicators (bool): Whether to include death indicators.
            direction (bool): Whether to include direction indicators.
            clear_path_pixels (bool): Whether to include clear path pixels.
            length_aware (bool): Whether to include snake length.
            reward_params (dict): The reward parameters for compute reward.
            failed_init_val (float): Value for failed game initialization.
            close_food_episodes_skip (int): Increase the 'max_food_distance' by 1 every 'close_food_episodes_skip' episodes.
            close_food (int): number of episodes to use close food logic.
            max_init_len (int): maximum initiation length in a random start.
            increasing_start_len (bool): Whether to increase the staring length as the sanke gets better
            max_not_eaten (int): maximum number of steps with no eaten fruits before reset
        """
        self.game = game
        self.last_score = 0
        self.episode = -1
        self.steps = 0
        self.not_eaten_steps = 0
        self.val_mode = False
        self.last_start_prob = self.game.default_start_prob
        self.close_food = config['close_food']
        self.close_food_episodes_skip = config['close_food_episodes_skip']
        self.max_init_len = config['max_init_len']
        self.failed_init_val = config['failed_init_val']
        self.last_action = self.game.snake_direction
        self.reward_params = config['reward_params']
        self.preprocessor = Preprocessor(game, config)
        self.increasing_start_len = config['increasing_start_len']
        self.max_not_eaten = config['max_not_eaten']
        self.clear_path = True

    def is_clear_path_between_head_and_tail(self):
        """Check if there is a clear path between the snake's head and tail.

        Returns:
            bool: True if there is a clear path, False otherwise.
        """
        head_x, head_y = self.game.snake[0]
        tail_x, tail_y = self.game.snake[-1]
        snake_positions = set((x, y) for x, y in self.game.snake[:-1])
        visited = set()
        queue = deque([(head_x, head_y)])
        while queue:
            (x, y) = queue.popleft()
            if (x, y) == (tail_x, tail_y):
                return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited and (nx, ny) not in snake_positions:
                    if 0 <= nx < self.game.width and 0 <= ny < self.game.height:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def get_score(self):
        return self.game.score

    def handle_clear_path_reward(self, reward, snake_len):
        clear_path_reward = self.reward_params.get('no_clear_path', 0)
        clear_path_reward_len_dep = self.reward_params.get('no_clear_path_length_dependent', 0)
        if clear_path_reward != 0 or clear_path_reward_len_dep != 0:
            if not self.is_clear_path_between_head_and_tail():
                if self.clear_path: # check if the no clear path is new.
                    reward += clear_path_reward + clear_path_reward_len_dep*snake_len
                    self.clear_path = False
            else:
                self.clear_path = True
        return reward

    def compute_reward(self):
        """
        Compute the reward for the current state.

        Returns:
            float: The computed reward.
        """
        if self.steps <= 1 and self.game.game_over:
            return self.failed_init_val
        done = self.game.game_over
        move = self.last_action != self.game.snake_direction
        snake_len = len(self.game.snake)
        reward = self.reward_params['death'] * done
        reward += self.reward_params['move'] * move
        reward += (self.game.score - self.last_score) * self.reward_params['food']
        reward += (self.game.score - self.last_score) * self.reward_params.get('food_length_dependent', 0) * snake_len
        reward += done * self.reward_params.get('death_length_dependent', 0) * snake_len
        reward = self.handle_clear_path_reward(reward, snake_len)
        self.last_score = self.game.score
        return reward

    def preprocess_state(self):
        """
        Preprocess the state of the game.

        Returns:
            np.array: The preprocessed state.
        """
        return self.preprocessor.preprocess_state()

    def update_eaten_status(self):
        """
        Updates 'not_eaten_steps'

        Args:
            last_score (int): the last recorded score of the snake

        Returns:
            truncated (bool): Whether the game was cut short
        """
        truncated = False
        eaten = self.game.score > self.last_score
        if not eaten:
            self.not_eaten_steps +=1
        else:
            self.not_eaten_steps = 0
        if self.not_eaten_steps > self.max_not_eaten:
            truncated = True
        return truncated

    def step(self, action):
        """
        Execute a step in the game.

        Args:
            action (int): The action to be taken.

        Returns:
            tuple: The new state, reward, and done flag.
        """
        self.game.change_direction(self.preprocessor.postprocess_action(action))
        score, done = self.game.move()
        state = self.preprocess_state()
        truncated = self.update_eaten_status()
        reward = self.compute_reward()
        self.steps += 1
        if done and self.val_mode:
            self.game.default_start_prob = self.last_start_prob
        return state, reward, done, truncated, {"score": score}  # observation, reward, terminated, truncated, info

    def init_game_max_food_distance(self):
        if self.episode > self.close_food:
            self.game.max_food_distance = None
        else:
            self.game.max_food_distance = self.episode // self.close_food_episodes_skip + 1

    def validation_rest(self):
        self.val_mode = True
        self.last_start_prob = self.game.default_start_prob
        self.game.default_start_prob = 1
        self.game.max_food_distance = None
        self.game.reset_game()
        self.game.default_start_prob = self.last_start_prob

    def train_reset(self):
        self.val_mode = False
        self.episode += 1
        self.init_game_max_food_distance()
        self.game.reset_game(random.randint(2, self.max_init_len))

    def reset(self, options={"validation": False}):
        self.steps = 0
        self.last_score = 0
        self.not_eaten_steps = 0
        if options.get('validation', False):
            self.validation_rest()
        else:
            self.train_reset()
        state = self.preprocess_state()
        return state, None # state, info

    def on_validation_end(self, episode, rewards, scores):
        mean_score = np.nanmean(scores)
        if self.increasing_start_len:
            self.max_init_len = np.max([int(mean_score*1.5)+1, 2, self.max_init_len])