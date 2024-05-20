import time
import keyboard
from snake_game import SnakeGame  # Make sure to save the SnakeGame class code in a file named snake_game.py

def play_game():
    game = SnakeGame()
    keyboard.on_press_key("up", lambda _: game.change_direction('UP'))
    keyboard.on_press_key("down", lambda _: game.change_direction('DOWN'))
    keyboard.on_press_key("left", lambda _: game.change_direction('LEFT'))
    keyboard.on_press_key("right", lambda _: game.change_direction('RIGHT'))

    while not game.game_over:
        score, game_over = game.move()
        print("\033c", end="")
        print(f"Score: {score}")
        print_game_state(game)
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