import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
import numpy as np
from modeling.A2C import A2CAgent
import yaml

class PPOAgent(A2CAgent):
    def __init__(self, config_path, value_network, actor_network):
        super().__init__(config_path, value_network, actor_network)
        if isinstance(config_path, str):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)['trainer']
        else:
            config = config_path

        self.clip_epsilon = float(config["clip_epsilon"])
        self.value_loss_coef = float(config["value_loss_coef"])
        self.policy_loss_coef = float(config["policy_loss_coef"])

    def optimize_model(self, observations, actions, returns, advantages, old_log_probs):
        actions, returns, advantages, observations, old_log_probs = self.prepare_tensors(actions, returns, advantages,
                                                                                         observations, old_log_probs)
        value_loss = self.optimize_value_network(observations, returns)
        actor_loss, entropy = self.optimize_actor_network(observations, actions, advantages, old_log_probs)
        self.debugger.track_loss(actor_loss, value_loss)
        self.debugger.track_policy_entropy(entropy.cpu().detach().numpy())

    def prepare_tensors(self, actions, returns, advantages, observations, old_log_probs):
        actions = F.one_hot(torch.tensor(actions, dtype=torch.int64), self.n_actions).float().to(self.device)
        returns = torch.tensor(returns[:, None], dtype=torch.float).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        observations = torch.tensor(observations, dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float).to(self.device)
        return actions, returns, advantages, observations, old_log_probs

    def optimize_actor_network(self, observations, actions, advantages, old_log_probs):
        self.actor_optimizer.zero_grad()
        policies = self.actor_network(observations)
        actor_loss, entropy = self.compute_actor_loss_entropy(policies, actions, advantages, old_log_probs)
        actor_loss.backward()
        self.debugger.track_gradients(True)
        self.actor_optimizer.step()
        return actor_loss, entropy

    def compute_actor_loss_entropy(self, policies, actions, advantages, old_log_probs):
        probs, log_probs, log_action_probs = self.compute_probabilities(policies, actions)
        ratio = torch.exp(log_action_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy = -(probs * log_probs).sum(-1).mean()
        actor_loss -= self.entropy_coefficient * entropy
        return actor_loss, entropy

    def update_model(self, observations, actions, rewards, dones, values, obs_torch, old_log_probs):
        next_value = [0] if dones[-1] else self.value_network(obs_torch).cpu().detach().numpy()[0]
        returns, advantages = self._returns_advantages(rewards, dones, values, next_value)
        self.optimize_model(observations, actions, returns, advantages, old_log_probs)
