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
        {'in_channels': 11, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    ]
    fc_layers = [512]

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

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_ncpp.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_closefood5005k_nodeathind.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_closefood5005k_nodir.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_closefood5005k_nofooddir.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_closefood5005k_nolenaware.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_closefood5005k_nonumeric.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_lendepreward.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_per06.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_mil20.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_ncp1.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_incslFalse.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_death5.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_lr5e3.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_lr5e5.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_lr1e5_ncp2_death3.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_normadv.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_ent001.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_ent05.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_bs512.yaml") # kaggle
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_ppo_ncp2_death3.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_ncp2_death3.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)