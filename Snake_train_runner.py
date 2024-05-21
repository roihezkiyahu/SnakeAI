from SnakeGame.snake_game import SnakeGame
from modeling.models import CNNDQNAgent
from modeling.trainer import Trainer
from modeling.SnakeWrapper import SnakeGameWrap
import copy
import os
from modeling.game_viz import GameVisualizer, GameVisualizer_cv2


if __name__ == "__main__":
    conv_layers_params = [
        {'in_channels': 11, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    ]
    output_size = 4
    game = SnakeGame(10,10,10, default_start_prob=1)
    model = CNNDQNAgent((11, game.width+2, game.height+2), output_size, dueling=True,
                        conv_layers_params=conv_layers_params)
    clone_model = CNNDQNAgent((11, game.width+2, game.height+2), output_size, dueling=True,
                              conv_layers_params=conv_layers_params)
    clone_model.load_state_dict(copy.deepcopy(model.state_dict()))

    game_wrapper = SnakeGameWrap(game, reward_params={'death': -1.5, 'move': 0, 'food': 1,
                                                      'food_length_dependent': 0, 'death_length_dependent': 0})

    visualizer = GameVisualizer_cv2(game)

    trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=5e-5,
                      gamma=0.99, validate_every_n_episodes=100,
                     folder=os.path.join("logging", "morechannels_lessDimReduction_vistest"), save_gif_every_x_epochs=100,
                     max_episode_len=10000, n_memory_episodes=100,
                     use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512,
                      EPS_DECAY=250, replaymemory=5000, game_wrapper=game_wrapper, visualizer=visualizer)
    trainer.train()