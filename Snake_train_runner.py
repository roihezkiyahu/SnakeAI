from SnakeGame.snake_game import SnakeGame
from modeling.models import CNNDQNAgent, ActorCritic
from modeling.trainer import Trainer
from modeling.SnakeWrapper import SnakeGameWrap
import copy
import os
from modeling.game_viz import GameVisualizer, GameVisualizer_cv2
from modeling.A2C import A2CAgent
from Atari_runner import (load_config, initialize_game, get_game_wrapper, train_a2c, train_ppo,
                          create_models, initialize_trainer)


def train_agent(config_path, conv_layers_params, fc_layers, continuous=None, a2c=False,
                game_wrapper=None, game=None, ppo=False):
    config = load_config(config_path)
    conv_layers_params = conv_layers_params.copy() if not conv_layers_params is None else None
    if game is None:
        game = initialize_game(config, continuous)
    game_wrapper = get_game_wrapper(game, config, game_wrapper)
    config['trainer']['game_wrapper'] = game_wrapper
    dueling = config['trainer'].get("dueling", True)
    use_cnn = config['trainer'].get("use_cnn", True)
    if config['trainer']["visualizer"] == "snake":
        config['trainer']["visualizer"] = GameVisualizer_cv2(game)
    if a2c or config["trainer"].get("a2c", False):
        train_a2c(config, game, game_wrapper, conv_layers_params, fc_layers)
        return
    if ppo or config["trainer"].get("ppo", False):
        train_ppo(config, game, game_wrapper, conv_layers_params, fc_layers)
        return

    model, clone_model = create_models(config, game_wrapper, conv_layers_params, fc_layers, dueling, use_cnn)
    trainer = initialize_trainer(config, model, clone_model)
    try:
        trainer.train()
    finally:
        pass

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

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_normadv.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_bs512.yaml") # kaggle
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_ent001.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)
    #
    # config_path = os.path.join("modeling", "configs", "trainer_config_snake_ppo_ncp2_death3.yaml")
    # train_agent(config_path, conv_layers_params, fc_layers,
    #             game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_cpp_ncp2_death3.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)

    config_path = os.path.join("modeling", "configs", "trainer_config_snake_a2c_ncp2_death3_ent05.yaml")
    train_agent(config_path, conv_layers_params, fc_layers,
                game=SnakeGame(10, 10, 10, default_start_prob=0.1), game_wrapper=SnakeGameWrap)