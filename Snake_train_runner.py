from SnakeGame.snake_game import SnakeGame
from modeling.models import CNNDQNAgent, ActorCritic
from modeling.trainer import Trainer
from modeling.SnakeWrapper import SnakeGameWrap
import copy
import os
from modeling.game_viz import GameVisualizer, GameVisualizer_cv2
from modeling.A2C import A2CAgent


if __name__ == "__main__":

    ## regular run
    # conv_layers_params = [
    #     {'in_channels': 11, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    #     {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    #     {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1}
    # ]
    # output_size = 4
    # game = SnakeGame(10,10,10, default_start_prob=1)
    # model = CNNDQNAgent((11, game.width+2, game.height+2), output_size, dueling=True,
    #                     conv_layers_params=conv_layers_params)
    # clone_model = CNNDQNAgent((11, game.width+2, game.height+2), output_size, dueling=True,
    #                           conv_layers_params=conv_layers_params)
    # clone_model.load_state_dict(copy.deepcopy(model.state_dict()))
    #
    # game_wrapper = SnakeGameWrap(game, reward_params={'death': -1.5, 'move': 0, 'food': 1,
    #                                                   'food_length_dependent': 0, 'death_length_dependent': 0})
    #
    # visualizer = GameVisualizer_cv2(game)
    #
    # trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=5e-5,
    #                   gamma=0.99, validate_every_n_episodes=100,
    #                  folder=os.path.join("logging", "morechannels_lessDimReduction_vistest"), save_gif_every_x_epochs=100,
    #                  max_episode_len=10000, n_memory_episodes=100,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512,
    #                   EPS_DECAY=250, replaymemory=5000, game_wrapper=game_wrapper, visualizer=visualizer,
    #                   reset_options={"validation": False})
    # trainer.train()


    ### A2C run
    conv_layers_params = [
        {'in_channels': 11, 'out_channels': 11, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups':11},
        {'in_channels': 11, 'out_channels': 16, 'kernel_size': 1, 'stride': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ]
    fc_layers = [256]


    game = SnakeGame(10, 10, 10, default_start_prob=0.1)

    input_shape = (11, game.width + 2, game.height + 2)
    action_size = 4
    actor_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='actor',
                              use_instance_norm=True)
    critic_model = ActorCritic(input_shape, action_size, conv_layers_params, fc_layers, mode='critic',
                               use_instance_norm=True)
    game_wrapper = SnakeGameWrap(game, reward_params={'death': -1.5, 'move': 0, 'food': 1,
                                                      'food_length_dependent': 0, 'death_length_dependent': 0})

    visualizer = GameVisualizer_cv2(game)

    A2C = A2CAgent(game, critic_model, actor_model, folder=os.path.join("logging", "A2CIN_bs256_wrapper_test_1e4"),
                   value_network_lr=1e-4, actor_network_lr=1e-4,
                   validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=250,
                   max_episode_len=5000, input_shape=input_shape, n_memory_episodes=250, game_wrapper=game_wrapper,
                   visualizer=visualizer, reset_options={"validation": False}, save_diagnostics=100)

    A2C.training_batch(150000, 256)