import time
import keyboard
from snake_game import SnakeGame  # Make sure to save the SnakeGame class code in a file named snake_game.py

def play_game():
    game = SnakeGame()

    # Set up key press events to change direction
    keyboard.on_press_key("up", lambda _: game.change_direction('UP'))
    keyboard.on_press_key("down", lambda _: game.change_direction('DOWN'))
    keyboard.on_press_key("left", lambda _: game.change_direction('LEFT'))
    keyboard.on_press_key("right", lambda _: game.change_direction('RIGHT'))

    while not game.game_over:
        # Move the snake
        score, game_over = game.move()

        # Clear the screen and print the game state
        print("\033c", end="")  # This escape sequence clears the screen in most terminals
        print(f"Score: {score}")
        print_game_state(game)

        # Adjust the sleep time to control the speed of the game
        time.sleep(0.2)


def print_game_state(game):
    for y in range(game.height):
        for x in range(game.width):
            if (x, y) == game.food:
                print("F", end="")
            elif (x, y) in game.snake:
                print("S", end="")
            else:
                print(".", end="")
        print()


if __name__ == "__main__":
    play_game()
