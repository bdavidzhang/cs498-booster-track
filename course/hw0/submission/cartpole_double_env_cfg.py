"""HW0 — YOUR TASK: double cartpole (two-link pendulum), balance and swingup.

Build this the way `cartpole_env_cfg.py` is built. The physical system is `cartpole_double.xml`:
the same cart and rail, but the pole is now two links. Open the XML and
note the joint names, which link each hinge belongs to, and that `hinge_2` is defined *relative to* `pole_1`, not relative to the world.

Two task variants share one config function, exactly like the cartpole:
  * Balance: start with both links up, keep them up.
  * Swingup: start hanging down, swing up and balance.

Provided for you (do NOT modify):
  * `joint_velocity_limit_exceeded` = a safety termination. The double
    pendulum is chaotic, and runaway joint velocities can blow up the
    fixed-timestep integrator; this guard ends those episodes cleanly.
  * Both PPO runner configs at the bottom — everyone trains with the same
    settings, so results are comparable and the autograder can reproduce
    your run. Do not tune them.

Your work is the three TODOs. The registration in hw0/__init__.py and all
public function names/signatures must not change (the autograder imports
them). Grading: see hw0/README.md.
"""

from __future__ import annotations

import math  # noqa: F401  (you will want it)
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  joint_pos_rel,  # noqa: F401
  joint_vel_rel,  # noqa: F401
  reset_joints_by_offset,  # noqa: F401
  time_out,  # noqa: F401
)
from mjlab.envs.mdp.actions import JointEffortActionCfg  # noqa: F401
from mjlab.managers.action_manager import ActionTermCfg  # noqa: F401
from mjlab.managers.event_manager import EventTermCfg  # noqa: F401
from mjlab.managers.observation_manager import (  # noqa: F401
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: F401
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg  # noqa: F401
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg  # noqa: F401
from mjlab.sim import MujocoCfg, SimulationCfg  # noqa: F401
from mjlab.terrains import TerrainEntityCfg  # noqa: F401
from mjlab.viewer import ViewerConfig  # noqa: F401

from course_tasks.hw0.cartpole_env_cfg import (
  _gaussian_tolerance,  # noqa: F401
  _quadratic_tolerance,  # noqa: F401
  cartpole_ppo_runner_cfg,
  pole_angle_cos_sin,  # noqa: F401  (works for MULTIPLE joints — check its shape!)
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_CARTPOLE_DOUBLE_XML: Path = Path(__file__).parent / "cartpole_double.xml"
_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE1_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_1",))
_HINGE2_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_2",))

# Entity.


def _get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(_CARTPOLE_DOUBLE_XML))


_CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
)

_BALANCE_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": 0.0, "hinge_2": 0.0},
  joint_vel={".*": 0.0},  # regexes over joint names are allowed
)
_SWINGUP_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider": 0.0, "hinge_1": math.pi, "hinge_2": 0.0},
  joint_vel={".*": 0.0},
)


def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec, 
    articulation=_CARTPOLE_ARTICULATION,
    init_state=_SWINGUP_INIT if swing_up else _BALANCE_INIT,
  ) 


# Rewards.




def cartpole_double_smooth_reward(
  env: ManagerBasedRlEnv,
  cart_cfg: SceneEntityCfg = _CART_CFG,
  hinge1_cfg: SceneEntityCfg = _HINGE1_CFG,
  hinge2_cfg: SceneEntityCfg = _HINGE2_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[cart_cfg.name]

  hinge1_angle = asset.data.joint_pos[:, hinge1_cfg.joint_ids].squeeze(-1)
  hinge2_angle = asset.data.joint_pos[:, hinge2_cfg.joint_ids].squeeze(-1)
  upright = ((torch.cos(hinge1_angle)  + 1) / 2 + (torch.cos(hinge1_angle+hinge2_angle) + 1) / 2) / 2

  cart_pos = asset.data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
  centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

  control = env.action_manager.action.squeeze(-1)
  small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

  hinge1_vel = asset.data.joint_vel[:, hinge1_cfg.joint_ids].squeeze(-1)
  hinge2_vel = asset.data.joint_vel[:, hinge2_cfg.joint_ids].squeeze(-1)
  small_velocity = (1 + _gaussian_tolerance(hinge1_vel, margin=5.0)*_gaussian_tolerance(hinge2_vel,margin=5.0)) / 2

  return upright * centered * small_control * small_velocity

# Terminations. (PROVIDED — do not modify.)


def joint_velocity_limit_exceeded(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  limit: float,
) -> torch.Tensor:
  """Terminate if any of the selected joints' velocity exceeds `limit`.

  Safety guard against the double pendulum's chaotic dynamics driving joint
  velocity into the regime where mujoco-warp's fixed-timestep integrator
  diverges. `limit` is set far above anything a normal swing-up/balance
  needs, so this only fires on genuinely runaway states.
  """
  asset: Entity = env.scene[asset_cfg.name]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return (vel.abs() > limit).any(dim=-1)


# Environment config.


def _make_env_cfg(swing_up: bool = False) -> ManagerBasedRlEnvCfg:
  # TODO(3): assemble, following _make_env_cfg in cartpole_env_cfg.py.
  #  - observations: cart pos/vel, and (cos, sin) + velocity for BOTH
  #    hinges. You do not need a new observation function — reuse
  #    `pole_angle_cos_sin` with a SceneEntityCfg selecting both hinge
  #    joints, and check the resulting shape.
  #  - actions: effort on the slider actuator (unchanged from cartpole).
  #  - events: reset offsets for the slider AND both hinges (the cartpole
  #    only had one hinge to jitter; the slider range still depends on
  #    swing_up the same way).
  #  - rewards: your cartpole_double_smooth_reward.
  #  - terminations: time_out, PLUS the provided
  #    joint_velocity_limit_exceeded over both hinges with limit=50.0.
  #  - scene: entity name must be "cartpole" (the SceneEntityCfgs above use
  #    it); num_envs=1024, env_spacing=4.0, plane terrain.
  #  - viewer/sim/decimation/episode_length_s: same as the cartpole, but
  #    distance=5.0 frames both links better.
  cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))
  hinge1_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1",))
  hinge2_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_2",))
  hinges_cfg = SceneEntityCfg("cartpole", joint_names=("hinge_1", "hinge_2",))


  # Observations: named terms, concatenated in order. "actor" is what the
  # policy sees (with noise/corruption during training); "critic" can see a
  # privileged, clean copy. Here they're identical.
  actor_terms = {
    "cart_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": cart_cfg}),
    "pole_angle": ObservationTermCfg(
      func=pole_angle_cos_sin, params={"asset_cfg": hinges_cfg}
    ),
    "cart_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": cart_cfg}),
    "pole_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": hinges_cfg}),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}),
  }

  # Actions: one scalar effort on the cart's slide actuator. The policy
  # outputs in roughly [-1, 1]; scale maps that to actuator units.
  actions: dict[str, ActionTermCfg] = {
    "effort": JointEffortActionCfg(
      entity_name="cartpole",
      actuator_names=("slider",),
      scale=1.0,
    ),
  }

  # Events with mode="reset" run at every episode reset. Randomizing the
  # initial state (a small offset around init_state) is what stops the
  # policy from memorizing one trajectory.
  slider_range = (-0.1, 0.1) if not swing_up else (0.0, 0.0)
  events = {
    "reset_slider": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": slider_range,
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("slider",)),
      },
    ),
    "reset_hinge": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.034, 0.034),
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg("cartpole", joint_names=("hinge_1","hinge_2")),
      },
    ),
  }

  rewards = {
    "smooth_reward": RewardTermCfg(
      func=cartpole_double_smooth_reward,
      weight=1.0,
      params={"cart_cfg": cart_cfg, "hinge1_cfg": hinge1_cfg, "hinge2_cfg": hinge2_cfg},
    ),
  }

  # time_out=True marks a truncation (episode ran out of time) rather than a
  # failure — PPO bootstraps the value function differently for the two.
  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "joint_velocity_limit_exceeded": TerminationTermCfg(func=joint_velocity_limit_exceeded,params={"asset_cfg":hinges_cfg,"limit":50.0})
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"cartpole": _get_cartpole_cfg(swing_up=swing_up)},
      num_envs=1024,  # overridden at train time: --env.scene.num-envs 4096
      env_spacing=4.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="cartpole",
      body_name="cart",
      distance=5.0,
      elevation=-15.0,
      azimuth=0.0,
    ),
    sim=SimulationCfg(
      # Contacts disabled: nothing in this scene needs them, and it's faster.
      mujoco=MujocoCfg(timestep=0.01, disableflags=("contact",)),
    ),
    # Policy acts every `decimation` physics steps: control dt = 0.05 s.
    decimation=5,
    episode_length_s=50.0,
  )


def cartpole_double_balance_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=False)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


def cartpole_double_swingup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg(swing_up=True)
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg


# RL config. (PROVIDED, do not modify)


def cartpole_double_balance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = cartpole_ppo_runner_cfg()
  cfg.experiment_name = "cartpole_double"
  cfg.max_iterations = 1500
  return cfg


def cartpole_double_swingup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = cartpole_ppo_runner_cfg()
  cfg.experiment_name = "cartpole_double"
  cfg.max_iterations = 5000
  return cfg
