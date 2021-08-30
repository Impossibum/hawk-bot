import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    VelocityPlayerToBallReward,
)
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import math as rl_math
from bubo_misc_utils import sign, normalize, distance, distance2D
import numpy as np

SIDE_WALL_X = 4096  # +/-
BACK_WALL_Y = 5120  # +/-
CEILING_Z = 2044
BACK_NET_Y = 6000  # +/-

GOAL_HEIGHT = 642.775

ORANGE_GOAL_CENTER = (0, BACK_WALL_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_CENTER = (0, -BACK_WALL_Y, GOAL_HEIGHT / 2)

# Often more useful than center
ORANGE_GOAL_BACK = (0, BACK_NET_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_BACK = (0, -BACK_NET_Y, GOAL_HEIGHT / 2)

BALL_RADIUS = 92.75

BALL_MAX_SPEED = 6000

BLUE_TEAM = 0
ORANGE_TEAM = 1


class SaveBoostReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        return math.sqrt(player.boost_amount) / 10


class KickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.vel_dir_reward = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if state.ball.position[0] == 0 and state.ball.position[1] == 0:
            return self.vel_dir_reward.get_reward(player, state, previous_action)
        return 0


class ThreeManRewards(RewardFunction):
    def __init__(self, min_spacing: float = 1500):
        super().__init__()
        self.min_spacing = min_spacing
        self.role_one_rewards = [
            VelocityBallToGoalReward(),
            KickoffReward(),
            TouchBallReward(),
        ]

    def spacing_reward(self, player: PlayerData, state: GameState, role: int):
        reward = 0
        if role != 0:
            for p in state.players:
                if p.team_num == player.team_num and p.car_id != player.car_id:
                    separation = distance(player.car_data.position, p.car_data.position)
                    if separation < self.min_spacing:
                        reward -= 1 - (separation / self.min_spacing)
        return reward

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        player_distances = []
        for p in state.players:
            if p.team_num == player.team_num:
                player_distances.append(
                    (distance(p.car_data.position, state.ball.position), p.car_id)
                )

        role = 0
        player_distances.sort(key=lambda x: x[0])
        for count, pd in enumerate(player_distances):
            if pd[1] == player.car_id:
                role = count
                break

        reward = self.spacing_reward(player, state, role)
        if role == 1:
            for rew in self.role_one_rewards:
                reward += rew.get_reward(player, state, previous_action)

        return reward


class NaiveSpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        linear_velocity = player.car_data.linear_velocity
        return rl_math.vecmag(linear_velocity) / 2300


class TouchBallReward(RewardFunction):
    def __init__(
        self,
        min_touch: float = 0.05,
        min_height: float = 160,
        aerial_weight: float = 0.15,
    ):
        self.min_touch = min_touch
        self.min_height = min_height
        self.aerial_weight = aerial_weight

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched:
            if state.ball.position[2] >= self.min_height:
                return max(
                    [
                        self.min_touch,
                        (
                            abs(state.ball.position[2] - BALL_RADIUS)
                            ** self.aerial_weight
                        )
                        - 1,
                    ]
                )

        return 0


class VelocityBallToGoalReward(RewardFunction):
    def __init__(self, own_goal=False, use_scalar_projection=False):
        super().__init__()
        self.own_goal = own_goal
        self.use_scalar_projection = use_scalar_projection

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.team_num == BLUE_TEAM
            and state.ball.position[1] > 0
            or player.team_num == ORANGE_TEAM
            and state.ball.position[1] < 0
        ):
            if (
                player.team_num == BLUE_TEAM
                and not self.own_goal
                or player.team_num == ORANGE_TEAM
                and self.own_goal
            ):
                if abs(state.ball.position[0]) < 800:
                    objective = np.array(ORANGE_GOAL_BACK)
                else:
                    objective = np.array(ORANGE_GOAL_CENTER)
            else:
                if abs(state.ball.position[0]) < 800:
                    objective = np.array(BLUE_GOAL_BACK)
                else:
                    objective = np.array(BLUE_GOAL_CENTER)

            vel = state.ball.linear_velocity
            pos_diff = objective - state.ball.position
            if self.use_scalar_projection:
                inv_t = rl_math.scalar_projection(vel, pos_diff)
                return inv_t
            else:
                norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
                vel /= BALL_MAX_SPEED
                return float(np.dot(norm_pos_diff, vel))
        return 0


class VelocityBallDefense(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if (
            player.team_num == BLUE_TEAM
            and state.ball.position[1] < 0
            or player.team_num == ORANGE_TEAM
            and state.ball.position[1] > 0
        ):
            if player.team_num == BLUE_TEAM:
                defense_objective = np.array(BLUE_GOAL_BACK)
            else:
                defense_objective = np.array(ORANGE_GOAL_BACK)

            vel = state.ball.linear_velocity
            pos_diff = state.ball.position - defense_objective
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            vel /= BALL_MAX_SPEED
            return float(np.dot(norm_pos_diff, vel))
        return 0


class PositioningReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.team_num == BLUE_TEAM:
            ball = state.ball.position
            pos = player.car_data.position
        else:
            ball = state.inverted_ball.position
            pos = player.inverted_car_data.position

        reward = 0.0
        if ball[1] < pos[1]:
            diff = abs(ball[1] - pos[1])
            reward = -(diff / 12000)
        return reward


if __name__ == "__main__":
    pass
