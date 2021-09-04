import rlgym
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    VelocityPlayerToBallReward,
    FaceBallReward,
)
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward
from hawk_rewards import *
from hawk_obs import AdvancedObsRevised3 as AdvancedObsRevised
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    NoTouchTimeoutCondition,
    GoalScoredCondition,
)
from rlgym.utils.state_setters import DefaultState
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_multidiscrete_wrapper import SB3MultiDiscreteWrapper
from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
from torch.nn import ReLU

frame_skip = 8
half_life_seconds = 3
num_instances = 6
team_size = 3
fps = 120 / frame_skip
gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
intended_batch_size = 10_000
steps = int(intended_batch_size/(num_instances*2))
batch_size = steps*num_instances*2


def get_match():
    reward_one = CombinedReward(
        (
            VelocityPlayerToBallReward(),
            TouchBallReward(min_touch=0.5, aerial_weight=0.2),
        ),
        (1.0, 1.0),
    )
    reward_two = CombinedReward(
        (
            ThreeManRewards(),
            VelocityBallDefense(),
            PositioningReward(),
        ),
        (1.0, 1.0, 1.0),
    )
    reward_three = CombinedReward(
        (
            ThreeManRewards(),
            PositioningReward(),
            EventReward(
                goal=10.0,
                team_goal=10.0,
                concede=-5.0,
                touch=0.0,
                shot=5.0,
                save=5.0,
                demo=5.0,
            ),
            SaveBoostReward(),
        ),
        (1.0, 1.0, 1.0, 0.5),
    )

    return Match(
        team_size=team_size,
        tick_skip=frame_skip,
        reward_function=AnnealRewards(reward_one, 40_000_000, reward_two, 400_000_000, reward_three),
        self_play=True,
        terminal_conditions=[
            TimeoutCondition(fps * 300),
            NoTouchTimeoutCondition(fps * 20),
            GoalScoredCondition(),
        ],
        obs_builder=AdvancedObsRevised(),
        state_setter=DefaultState(),
    )


def model_training():
    epic_path = None
    num_procs = num_instances
    num_ts = 1_000_000_000

    env = SB3MultipleInstanceEnv(epic_path, get_match, num_procs, wait_time=22)
    env = VecCheckNan(env)
    env = SB3MultiDiscreteWrapper(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)
    env.reset()

    callback = CheckpointCallback(int(10_000_000 / env.num_envs), "hawk")

    try:
        custom_objects = {
            "n_envs": env.num_envs,
            "batch_size": batch_size,
            "n_steps": steps,
        }
        learner = PPO.load("hawk/exit_save", env=env, custom_objects=custom_objects)
    except:
        policy_kwargs = dict(
            activation_fn=ReLU,
            net_arch=[dict(pi=[1024, 512, 256, 256], vf=[1024, 512, 256, 256])],
        )
        learner = PPO(
            "MlpPolicy",
            env,
            n_epochs=50,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-5,
            ent_coef=0.01,
            gamma=gamma,
            vf_coef=1.0,
            verbose=3,
            batch_size=batch_size,
            n_steps=steps,
            tensorboard_log="hawk/logs",
            device="auto",
        )
    try:
        learner.learn(
            total_timesteps=num_ts, callback=callback, reset_num_timesteps=False
        )
        learner.save(f"hawk/{num_ts}_steps")
        print(f"{num_ts} steps reached!")
        print("exiting")
    except:
        learner.save("hawk/exit_save")
        print("exiting, model saved!")


if __name__ == "__main__":
    model_training()
