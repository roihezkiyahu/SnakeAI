from SnakeGame.snake_game import SnakeGame
from modeling.models import CNNDQNAgent, ActorCritic
from modeling.trainer import Trainer
from modeling.SnakeWrapper import SnakeGameWrap
import copy
import os
from modeling.game_viz import GameVisualizer, GameVisualizer_cv2
from modeling.A2C import A2CAgent
from Atari_runner import train_agent


if __name__ == "__main__":
    conv_layers_params = [
        {'in_channels': 11, 'out_channels': 11, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups': 11},
        {'in_channels': 11, 'out_channels': 16, 'kernel_size': 1, 'stride': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ]
    fc_layers = [256]

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_bs32.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_bs512.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_lendepreward.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_gamma90.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_noextra.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_ncpp.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_closefood5005k.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpponly.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_death_ind.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_fooddir.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_lenaware.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_numeric.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_snakedir.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_lendepreward.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_per06.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)


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