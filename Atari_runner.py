from modeling.models import CNNDQNAgent, DQN, ActorCritic
from modeling.trainer import Trainer
from modeling.AtariGameWrapper import AtariGameWrapper
import copy
import os
import gym
import gymnasium as gymnas
import ale_py
from modeling.A2C import A2CAgent
import yaml


def train_a2c(config, game, game_wrapper, conv_layers_params, fc_layers):
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game.action_space.n
    actor_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='actor')
    critic_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='critic')
    config["trainer"]["input_shape"] = input_shape
    config['trainer']['n_actions'] = output_size
    A2C = A2CAgent(config["trainer"], critic_model, actor_model)

    A2C.training_batch(config["trainer"]["epochs"], config["trainer"]["batch_size"])


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def initialize_game(config, continuous):
    if continuous is not None:
        game = gym.make(config['trainer']['game'], render_mode=config['trainer']['render_mode'],
                        continuous=continuous, frameskip=config['atari_game_wrapper']['frame_skip'])
    else:
        game = gym.make(config['trainer']['game'], render_mode=config['trainer']['render_mode'],
                        frameskip=config['atari_game_wrapper']['frame_skip'])
    return game


def get_game_wrapper(game, config, game_wrapper):
    if game_wrapper is None:
        game_wrapper = AtariGameWrapper(game, config['atari_game_wrapper'])
    else:
        game_wrapper = game_wrapper(game, config['game_wrapper'])
    return game_wrapper


def initialize_trainer(config, model, clone_model):
    return Trainer(config['trainer'], model, clone_model)


def create_models(config, game_wrapper, conv_layers_params, fc_layers, dueling, use_cnn=True):
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game_wrapper.game.action_space.n

    if use_cnn:
        model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=dueling)
        clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=dueling)
    else:
        n_observations = state.shape[0]
        model = DQN(fc_layers, n_observations, output_size)
        clone_model = DQN(fc_layers, n_observations, output_size)

    config['trainer']['n_actions'] = output_size
    return model, clone_model


def train_agent(config_path, conv_layers_params, fc_layers, dueling, continuous=None, a2c=False,
                game_wrapper=None, use_cnn=True):
    config = load_config(config_path)
    game = initialize_game(config, continuous)
    game_wrapper = get_game_wrapper(game, config, game_wrapper)
    config['trainer']['game_wrapper'] = game_wrapper
    if a2c:
        train_a2c(config, game, game_wrapper, conv_layers_params, fc_layers)
        return

    model, clone_model = create_models(config, game_wrapper, conv_layers_params, fc_layers, dueling, use_cnn)
    trainer = initialize_trainer(config, model, clone_model)
    try:
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    # # CarRacing A2C example
    # game = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    # conv_layers_params = [
    #     {'in_channels': 3, 'out_channels': 3, 'kernel_size': 5, 'stride': 1, 'padding': 1, 'groups':3},
    #     {'in_channels': 3, 'out_channels': 8, 'kernel_size': 1, 'stride': 1},
    #     {'in_channels': 8, 'out_channels': 16, 'kernel_size': 5, 'stride': 4, 'padding': 1},
    #     {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 4, 'padding': 1},
    #     {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    # ]
    # fc_layers = [256, 128]
    # state, info = game.reset()
    # input_shape, output_size = (state.shape[2], state.shape[0], state.shape[1]), game.action_space.n
    #
    # actor_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder=os.path.join("logging", "CarRacing_1e4_A2C"),
    #                value_network_lr=1e-4, actor_network_lr=1e-4,
    #                validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=5,
    #                max_episode_len=1000, input_shape=input_shape, n_memory_episodes=250
    #                , reset_options={"validation": False}, gif_fps=10, n_actions=output_size)
    #
    # A2C.training_batch(10000, 128)

    # SkiingDeterministic
    # game = gym.make("SkiingDeterministic-v4", render_mode="rgb_array")
    # conv_layers_params = [
    #     {'in_channels': 3, 'out_channels': 3, 'kernel_size': 5, 'stride': 1, 'padding': 1, 'groups':3},
    #     {'in_channels': 3, 'out_channels': 8, 'kernel_size': 1, 'stride': 1},
    #     {'in_channels': 8, 'out_channels': 16, 'kernel_size': 5, 'stride': 4, 'padding': 1},
    #     {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 4, 'padding': 1},
    #     {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    # ]
    # fc_layers = [256, 128]
    # state, info = game.reset()
    # input_shape, output_size = (state.shape[2], state.shape[0], state.shape[1]), game.action_space.n
    #
    # actor_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder=os.path.join("logging", "SkiingDeterministic_1e4_A2C"),
    #                value_network_lr=1e-4, actor_network_lr=1e-4,
    #                validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=50,
    #                max_episode_len=1000, input_shape=input_shape, n_memory_episodes=250
    #                , reset_options={"validation": False}, gif_fps=10, n_actions=output_size)
    #
    # A2C.training_batch(10000, 128)
    #
    # layer_params = [
    #     {'out_features': 256},
    #     {'out_features': 128},
    #     {'out_features': 64},
    # ]

    # game = gymnas.make("CartPole-v1", render_mode="rgb_array")
    # game = gymnas.make('MountainCar-v0', render_mode="rgb_array")
    #
    # output_size = game.action_space.n
    # state, info = game.reset()
    # n_observations = state.shape[0]
    #
    # model = DQN(layer_params, n_observations, output_size)
    # clone_model = DQN(layer_params, n_observations, output_size)
    #
    # trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=1e-3,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "MountainCar_1e3_disc05_rand"), save_gif_every_x_epochs=100,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512, EPS_DECAY=250,
    #                    max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.5,
    #                   reset_options={'randomize_position': True})
    #
    # trainer.train()

    # game = gymnas.make('Acrobot-v1', render_mode="rgb_array")
    #
    # output_size = game.action_space.n
    # state, info = game.reset()
    # n_observations = state.shape[0]
    #
    # model = DQN(layer_params, n_observations, output_size)
    # clone_model = DQN(layer_params, n_observations, output_size)
    #
    # trainer = Trainer(game, model, clone_model, episodes=1000, learning_rate=1e-4,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "Acrobot_1e4_disc075"), save_gif_every_x_epochs=100,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512, EPS_DECAY=250,
    #                    max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.75,
    #                   reset_options={'randomize_position': False}, update_every_n_steps=10)
    #
    # trainer.train()

    # game = gymnas.make("LunarLander-v2", render_mode="rgb_array")
    #
    # layer_params = [
    #     {'out_features': 64},
    #     {'out_features': 64},
    # ]
    # state, info = game.reset()
    # n_observations, output_size = state.shape[0], game.action_space.n
    #
    # model = DQN(layer_params, n_observations, output_size, dueling=True)
    # clone_model = DQN(layer_params, n_observations, output_size, dueling=True)
    #
    # trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=5e-4,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "LunarLander_5e4_disc099_dueling"), save_gif_every_x_epochs=50,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=64, EPS_DECAY=200,
    #                    max_episode_len=500, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.99,
    #                   reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=10)
    #
    # trainer.train()

    conv_layers_params = [
        {'in_channels': 4, 'out_channels': 8, 'kernel_size': 7, 'stride': 4, 'padding': 1},
        {'in_channels': 8, 'out_channels': 16, 'kernel_size': 7, 'stride': 4, 'padding': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}]
    fc_layers = [256, 128]


    # SpaceInvaders
    config_path = os.path.join("modeling", "configs", "trainer_config_SpaceInvaders.yaml")
    dueling = True
    train_agent(config_path, conv_layers_params, fc_layers, dueling)

    config_path = os.path.join("modeling", "configs", "trainer_config_SpaceInvaders_per06.yaml")
    dueling = True
    train_agent(config_path, conv_layers_params, fc_layers, dueling)

    config_path = os.path.join("modeling", "configs", "trainer_config_SpaceInvaders_gamma999.yaml")
    dueling = True
    train_agent(config_path, conv_layers_params, fc_layers, dueling)

    # MsPacmanNoFrameskip
    config_path = os.path.join("modeling", "configs", "trainer_config_MsPacman.yaml")
    dueling = True
    train_agent(config_path, conv_layers_params, fc_layers, dueling)

    # BreakoutNoFrameskip
    config_path = os.path.join("modeling", "configs", "trainer_config_Breakout.yaml")
    dueling = True
    train_agent(config_path, conv_layers_params, fc_layers, dueling)

    # SkiingDeterministic
    config_path = os.path.join("modeling", "configs", "trainer_config_Skiing.yaml")
    dueling = True
    train_agent(config_path, conv_layers_params, fc_layers, dueling)


    # CarRacing
    config_path = os.path.join("modeling", "configs", "trainer_config_CarRacing.yaml", continuous=False)
    dueling = True
    # conv_layers_params = [
    #     {'in_channels': 4, 'out_channels': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups':4},
    #     {'in_channels': 4, 'out_channels': 8, 'kernel_size': 1, 'stride': 1},
    #     {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 4, 'padding': 1},
    #     {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 4, 'padding': 1},
    #     {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1}]
    # fc_layers = [128]
    train_agent(config_path, conv_layers_params, fc_layers, dueling)
