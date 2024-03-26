import numpy as np


def initialize_state(game):
    """Initialize the state grid."""
    state = np.zeros((game.height, game.width), dtype=np.float32)
    return state


def mark_snake_on_state(game, state):
    """Mark the snake's body, head, and tail on the state grid."""
    # Mark the snake's body
    for x, y in game.snake[1:-1]:
        state[y, x] = 2
    # Mark the snake's tail
    tail_x, tail_y = game.snake[-1]
    state[tail_y, tail_x] = 3
    # Mark the snake's head
    head_x, head_y = game.snake[0]
    state[head_y, head_x] = 1
    return state


def mark_food_on_state(game, state):
    """Mark the food's position on the state grid."""
    food_x, food_y = game.food
    state[food_y, food_x] = 4
    return state


def make_border(game, state):
    new_state = np.full((game.height + 2, game.width + 2), 10, dtype=np.float32)
    new_state[1:-1, 1:-1] = state
    return new_state


def calculate_food_direction(game):
    """Calculate the direction and distance to the food from the snake's head."""
    head_x, head_y = game.snake[0]
    food_x, food_y = game.food
    left = food_x < head_x
    right = food_x > head_x
    up = food_y > head_y
    down = food_y < head_y
    #     radius = math.sqrt((food_x - head_x)**2 + (food_y - head_y)**2)
    #     theta = math.atan2(food_y - head_y, food_x - head_x)
    return np.array([left, right, up, down], dtype=np.float32)


def calculate_death_indicators(game):
    """Calculate indicators for whether moving left, right, up, or down would result in death."""
    head_x, head_y = game.snake[0]
    death_indicators = np.zeros(4, dtype=np.float32)  # [left, right, up, down]
    for i, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
        nx, ny = head_x + dx, head_y + dy
        if nx < 0 or nx >= game.width or ny < 0 or ny >= game.height or (nx, ny) in game.snake:
            death_indicators[i] = 1
    return death_indicators


def calculate_direction_indicators(game):
    direction = game.snake_direction
    return np.array([np.argmax(direction == ['UP', 'DOWN', 'LEFT', 'RIGHT'])], dtype=np.float32)


def preprocess_state(game, for_cnn=True, food_direction=True, add_death_indicators=True,
                     direction=True):
    """Convert the game state into a 2D grid suitable for CNN input, with optional features."""
    state = initialize_state(game)
    state = mark_snake_on_state(game, state)
    state = mark_food_on_state(game, state)
    state = make_border(game, state)

    additional_features = []

    if food_direction:
        additional_features.extend(calculate_food_direction(game))

    if add_death_indicators:
        additional_features.extend(calculate_death_indicators(game))

    if direction:
        additional_features.extend(calculate_direction_indicators(game))

    if for_cnn:
        state = state.reshape(game.height + 2, game.width + 2)
        if additional_features:
            additional_features_chanels = np.stack(
                [np.full((game.height + 2, game.width + 2), feat) for feat in additional_features] + [state], axis=0)
            return additional_features_chanels
    else:
        state = state.flatten()
        if additional_features:
            state = np.concatenate([state, additional_features])

    return state.astype(np.float32)


def postprocess_action(action):
    """Convert the neural network's output action into game action."""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    return actions[action]
