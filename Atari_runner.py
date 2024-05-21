from modeling.models import CNNDQNAgent, DQN
from modeling.trainer import Trainer
import copy
import os
import gymnasium as gym


if __name__ == "__main__":
    layer_params = [
        {'out_features': 256},
        {'out_features': 128},
        {'out_features': 64},
    ]

    game = gym.make("CartPole-v1", render_mode="rgb_array")
    # game = gym.make('MountainCar-v0', render_mode="rgb_array")
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

    # game = gym.make('Acrobot-v1', render_mode="rgb_array")
    #
    # output_size = game.action_space.n
    # state, info = game.reset()
    # n_observations = state.shape[0]
    #
    # model = DQN(layer_params, n_observations, output_size)
    # clone_model = DQN(layer_params, n_observations, output_size)
    #
    # trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=1e-4,
    #                   gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
    #                  folder=os.path.join("logging", "Acrobot_1e4_disc075"), save_gif_every_x_epochs=100,
    #                  use_ddqn=True, EPS_END=0, EPS_START=1, batch_size=512, EPS_DECAY=250,
    #                    max_episode_len=1000, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.75,
    #                   reset_options={'randomize_position': False})
    #
    # trainer.train()

    game = gym.make("LunarLander-v2", render_mode="rgb_array")

    output_size = game.action_space.n
    state, info = game.reset()
    n_observations = state.shape[0]

    model = DQN(layer_params, n_observations, output_size, dueling=True)
    clone_model = DQN(layer_params, n_observations, output_size, dueling=True)

    trainer = Trainer(game, model, clone_model, episodes=10000, learning_rate=1e-4,
                      gamma=0.99, validate_every_n_episodes=100, n_memory_episodes=100,
                     folder=os.path.join("logging", "LunarLander_1e4_disc075_dueling"), save_gif_every_x_epochs=50,
                     use_ddqn=True, EPS_END=0, EPS_START=0.000001, batch_size=512, EPS_DECAY=250,
                       max_episode_len=500, replaymemory=5000, n_actions=output_size, gif_fps=10, discount_rate=0.75,
                      reset_options={'randomize_position': False})

    trainer.train()
