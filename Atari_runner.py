from modeling.models import CNNDQNAgent, DQN, ActorCritic
from modeling.trainer import Trainer
from modeling.AtariGameWrapper import AtariGameWrapper
import copy
import os
import gym
import gymnasium as gymnas
import ale_py
from modeling.A2C import A2CAgent



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
    #     {'in_channels': 3, 'out_channels': 16, 'kernel_size': 5, 'stride': 4, 'padding': 1},
    #     {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 4, 'padding': 1},
    #     {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
    #     {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
    # ]
    # fc_layers = [256, 128]
    # state, info = game.reset()
    # input_shape, output_size = (state.shape[2], 105, 80), game.action_space.n
    #
    # actor_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='actor',
    #                           use_instance_norm=True)
    # critic_model = ActorCritic(input_shape, output_size, conv_layers_params, fc_layers, mode='critic',
    #                            use_instance_norm=True)
    # game_wrapper = AtariGameWrapper(game, resize_img=(80, 105))
    #
    # A2C = A2CAgent(game, critic_model, actor_model, folder=os.path.join("logging", "SkiingDeterministic_1e4_A2C_resize_disc07"),
    #                value_network_lr=1e-4, actor_network_lr=1e-4,
    #                validate_every_n_episodes=1000, validate_episodes=250, save_gif_every_x_epochs=50,
    #                max_episode_len=200, input_shape=input_shape, n_memory_episodes=250
    #                , reset_options={"validation": False}, gif_fps=10, n_actions=output_size, game_wrapper=game_wrapper,
    #                discount_rate = 0.7)
    #
    # A2C.training_batch(10000, 128)

    layer_params = [
        {'out_features': 256},
        {'out_features': 128},
        {'out_features': 64},
    ]

    # game = gymnas.make("CartPole-v1", render_mode="rgb_array")
    # game = gymnas.make('MountainCar-v0', render_mode="rgb_array")
    # game_wrapper = AtariGameWrapper(game, random_steps_range=(0, 100))
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
    #                  folder=os.path.join("logging", "CartPole_1e3_rand_start"), save_gif_every_x_epochs=100,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512, EPS_DECAY=250,
    #                    max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=10,
    #                   reset_options={'randomize_position': False}, game_wrapper=game_wrapper)
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

    # BreakoutNoFrameskip
    # game = gym.make("BreakoutDeterministic-v4", render_mode="rgb_array")
    conv_layers_params = [
        {'in_channels': 1, 'out_channels': 8, 'kernel_size': 9, 'stride': 4, 'padding': 1},
        {'in_channels': 8, 'out_channels': 16, 'kernel_size': 7, 'stride': 4, 'padding': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 5, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}]
    fc_layers = [256, 128]

    # state, info = game.reset()
    # input_shape, output_size = (state.shape[2], state.shape[0], state.shape[1]), game.action_space.n
    #
    # model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)
    # clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)
    #
    # trainer = Trainer(game, model, clone_model, episodes=1000, learning_rate=5e-4,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "BreakoutDeterministic_5e4_disc099_dueling"), save_gif_every_x_epochs=50,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=64, EPS_DECAY=200,
    #                    max_episode_len=500, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.99,
    #                   reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=25)
    #
    # trainer.train()

    conv_layers_params = [
        {'in_channels': 3, 'out_channels': 8, 'kernel_size': 5, 'stride': 4, 'padding': 1},
        {'in_channels': 8, 'out_channels': 16, 'kernel_size': 5, 'stride': 4, 'padding': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
        {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}]
    fc_layers = [256, 128]
    game = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")

    game_wrapper = AtariGameWrapper(game, gray_scale=False)
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game.action_space.n

    model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)
    clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)

    trainer = Trainer(game, model, clone_model, episodes=2500, learning_rate=1e-4,
                      gamma=0.99, validate_every_n_episodes=10, n_memory_episodes=10,
                      folder=os.path.join("logging", "SpaceInvadersNoFrameskip_test"),
                      save_gif_every_x_epochs=50,
                      use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=64, EPS_DECAY=200,
                      max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=25,
                      reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=25,
                      save_diagnostics=200, game_wrapper=game_wrapper)

    trainer.train()

    # SkiingDeterministic
    game = gym.make("SkiingDeterministic-v4", render_mode="rgb_array")
    game_wrapper = AtariGameWrapper(game, resize_img=(80, 105), default_start_prob=0.25,
                                    random_steps_range=(250, 1000), stack_n_frames=0)
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game.action_space.n


    model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)
    clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)

    trainer = Trainer(game, model, clone_model, episodes=2500, learning_rate=1e-2,
                      gamma=0.8, validate_every_n_episodes=100, n_memory_episodes=100,
                     folder=os.path.join("logging", "SkiingDeterministic_1e2_gamma08_dueling_clip10"),
                      save_gif_every_x_epochs=1,
                     use_ddqn=True, EPS_END=0.05, EPS_START=1, batch_size=64, EPS_DECAY=500,
                       max_episode_len=1500, replaymemory=5000, n_actions=output_size, gif_fps=30,
                      reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=25,
                      save_diagnostics=200, game_wrapper=game_wrapper, clip_grad=50)

    trainer.train()


    # SpaceInvadersNoFrameskip
    game = gym.make("SpaceInvadersDeterministic-v4", render_mode="rgb_array")
    game_wrapper = AtariGameWrapper(game, resize_img=(80, 105))
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game.action_space.n

    model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)
    clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)

    trainer = Trainer(game, model, clone_model, episodes=1000, learning_rate=1e-4,
                      gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
                     folder=os.path.join("logging", "SpaceInvadersDeterministic_1e4_disc07_dueling"),
                      save_gif_every_x_epochs=50,
                     use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=64, EPS_DECAY=200,
                       max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=25,
                      reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=25,
                      save_diagnostics=200, game_wrapper=game_wrapper)

    trainer.train()

    # MsPacmanNoFrameskip
    game = gym.make("MsPacmanDeterministic-v4", render_mode="rgb_array")
    game_wrapper = AtariGameWrapper(game, resize_img=(80, 105))
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game.action_space.n

    model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)
    clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)

    trainer = Trainer(game, model, clone_model, episodes=1000, learning_rate=5e-4,
                      gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
                     folder=os.path.join("logging", "MsPacmanDeterministic_5e4_disc099_dueling"), save_gif_every_x_epochs=50,
                     use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=64, EPS_DECAY=200,
                       max_episode_len=2000, replaymemory=5000, n_actions=output_size, gif_fps=10,
                      reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=10,
                      game_wrapper=game_wrapper)

    trainer.train()

    # CarRacing
    game = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=False)
    conv_layers_params = [
        {'in_channels': 1, 'out_channels': 3, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'groups':3},
        {'in_channels': 3, 'out_channels': 8, 'kernel_size': 1, 'stride': 1},
        {'in_channels': 8, 'out_channels': 16, 'kernel_size': 3, 'stride': 4, 'padding': 1},
        {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 4, 'padding': 1},
        {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1}]
    fc_layers = [128]
    game_wrapper = AtariGameWrapper(game, resize_img=(80, 105))
    state, info = game_wrapper.reset()
    input_shape, output_size = state.shape, game.action_space.n

    model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)
    clone_model = CNNDQNAgent(input_shape, output_size, conv_layers_params, fc_layers, dueling=True)

    trainer = Trainer(game, model, clone_model, episodes=1000, learning_rate=5e-4,
                      gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
                     folder=os.path.join("logging", "CarRacing_5e4_disc095_dueling"), save_gif_every_x_epochs=50,
                     use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=64, EPS_DECAY=200,
                       max_episode_len=2000, replaymemory=5000, n_actions=output_size, gif_fps=25,
                      reset_options={'randomize_position': False}, per_alpha=0, update_every_n_steps=25,
                      game_wrapper=game_wrapper)

    trainer.train()
