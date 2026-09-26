"""HW2 — YOUR TASK: turn this into PPO.

This file starts as an exact copy of `reinforce.py`, with the class renamed to
`PPO`. Until you change it, the `Course-HW2-*-PPO` tasks train exactly like
REINFORCE. When you are done,

    diff src/course_tasks/hw2/reinforce.py src/course_tasks/hw2/ppo.py

should be your PPO. What to change, and which parts of the PPO paper to read,
is in Part II of hw2/README.md.

Contract — the runner and the autograder rely on it:
  * keep the class name `PPO`;
  * keep the names and signatures of `__init__`, `act`, `process_env_step`,
    `compute_returns(last_obs)`, `compute_advantages`, `update`, `state_dict`
    and `load_state_dict`;
  * keep the keys `state_dict` saves, or checkpoints you already have stop
    loading in `uv run play`.
You may add methods, attributes, and new keys to the dict `update` returns.
"""

#freely change here 

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from course_tasks.hw2.modules import Critic, GaussianActor
from course_tasks.hw2.storage import RolloutStorage


class PPO:
  """
  Obejctive is 
  $$
  L^{PPO} = -L^{CLIP} + c_v L^{V} - c_e H(\pi)
  $$
  - the first term is the clip loss (we want to maximize) 
  - the second is the quad reconstruction loss between V(st) and the target value (how do we get this?)
  - the third term is the entropy for this distribution (how do we get this, how does this prevent sigma from decreasing?)
  """

  def __init__(
    self,
    actor: GaussianActor,
    critic: Critic,
    storage: RolloutStorage,
    gamma: float = 0.99, #decay rate for Gt 
    learning_rate: float = 1.0e-3, #alpha used for theta_t update
    entropy_coef: float = 0.005, #this is ce in the total loss
    value_loss_coef: float = 0.5, #the is cv in the total loss
    max_grad_norm: float = 1.0, #this is a limit to the grad norm scale
    baseline: str = "value", #baseline form value esitmate, V(st)
    normalize_advantage: bool = True, #is At normalized
    clip_param: float = 0.2, #epsilon for the clip objective
    lam: float = 0.95, #lambda 
    num_learning_epochs: int = 5, #metadata 
    num_mini_batches: int = 4,
    device: str | torch.device = "cpu",
  ) -> None:
    self.actor = actor
    self.critic = critic
    self.storage = storage
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.entropy_coef = entropy_coef
    self.value_loss_coef = value_loss_coef
    self.max_grad_norm = max_grad_norm
    self.baseline = baseline
    self.normalize_advantage = normalize_advantage
    self.device = device

    # PPO hyperparameters, passed in from the runner config (--agent.algorithm.*).
    # Nothing below uses them yet.
    self.clip_param = clip_param
    self.lam = lam
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches

    self.params = list(actor.parameters())
    #baselines are either 0 (vanilla) or based on the critic output
    if baseline == "value":
      self.params += list(critic.parameters())
    self.optimizer = torch.optim.Adam(self.params, lr=learning_rate)

  # -- Collection. ----------------------------------------------------------

  def act(self, obs: torch.Tensor) -> torch.Tensor:
    """Sample an action for every environment and stash what update() needs.

    Called inside `torch.inference_mode()` by the runner, so nothing produced
    here carries gradients. That is why `update()` re-scores the actions.
    """
    actions, log_probs = self.actor.act(obs)
    values = self.critic(obs) if self.baseline == "value" else torch.zeros_like(log_probs)
    self._transition = (obs, actions, log_probs, values) #a shorthand for storing these states
    return actions

  def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """Record the outcome of the action `act` just returned."""
    obs, actions, log_probs, values = self._transition
    self.storage.add_transition(obs, actions, log_probs, rewards, dones, values)

  # -- Learning. ------------------------------------------------------------

  def compute_returns(self, last_obs: torch.Tensor) -> None:
    """Discounted return-to-go into `storage.returns`, computed backwards.

    `last_obs` is the observation after the final collected step. REINFORCE
    never uses it: every rollout ends on a genuine terminal, so there is no
    "value of what comes after" to bootstrap. It is in the signature because an
    algorithm whose rollouts can stop mid-episode does need it.
    """
    st = self.storage
    running = torch.zeros(st.num_envs, device=st.returns.device)

    T = st.num_transitions_per_env
    V_st = self.critic(last_obs)
    running = V_st
    for t in reversed(range(T)):
        running = st.rewards[t] + self.gamma * (1.0 - st.dones[t]) * running  
        st.returns[t] = running

  def compute_advantages(self) -> None:
    """
    `storage.advantages` from returns, minus the baseline, optionally normalized.
    A^{GAE}_t = delta_t + gamma * lambda * A^{GAE}_{t+1}
    """
    gamma = self.gamma
    lam = self.lam

    st = self.storage
    if self.baseline == "value":
      T = st.num_transitions_per_env
      advantages = st.returns.clone()
      advantages -= st.values 
      advantages[:T] += gamma * st.values[1:]
      for t in reversed(range(T)):
        advantages[t] += gamma * lam * advantages[t+1]
    else:
      advantages = st.returns.clone()
    if self.normalize_advantage:
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    st.advantages = advantages

  def update(self) -> dict[str, float]:
    """One full-batch gradient step. Returns losses for logging."""
    obs, actions, _old_log_probs, returns, advantages = self.storage.flatten()

    # Re-score the stored actions under the CURRENT policy: this is the copy of
    # log pi that carries gradients. The stored `old_log_probs` came out of
    # inference_mode and cannot be backpropagated through.
    log_probs, entropy = self.actor.evaluate_actions(obs, actions)

    policy_loss = -(log_probs * advantages).mean()
    entropy_loss = -entropy.mean()
    if self.baseline == "value":
      value_loss = F.mse_loss(self.critic(obs), returns)
    else:
      value_loss = torch.zeros((), device=obs.device)

    loss = (
      policy_loss
      + self.value_loss_coef * value_loss
      + self.entropy_coef * entropy_loss
    )

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
    self.optimizer.step()

    return {
      "policy": policy_loss.item(),
      "value": value_loss.item(),
      "entropy": entropy.mean().item(),
    }

  # -- Plumbing. ------------------------------------------------------------

  def train_mode(self) -> None:
    self.actor.train()
    self.critic.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()

  def state_dict(self) -> dict:
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }

  def load_state_dict(self, loaded: dict, strict: bool = True) -> None:
    self.actor.load_state_dict(loaded["actor_state_dict"], strict=strict)
    self.critic.load_state_dict(loaded["critic_state_dict"], strict=strict)
    if "optimizer_state_dict" in loaded:
      self.optimizer.load_state_dict(loaded["optimizer_state_dict"])
