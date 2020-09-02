import gym
from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.gym_wrapper.envs.cube_env import ActionType

from rrc_simulation import TriFingerPlatform
from rrc_simulation import visual_objects
from rrc_simulation.tasks import move_cube
import numpy as np


class CubeEnv(cube_env.CubeEnv):

    def __init__(self,
                 initializer=None,
                 action_type=cube_env.ActionType.POSITION,
                 frameskip=50,
                 visualization=False,
                 sparse_reward = True):

        super().__init__(initializer,
                         action_type,
                         frameskip,
                         visualization)

        self.sparse_reward = sparse_reward

        spaces = TriFingerPlatform.spaces

        object_state_space = gym.spaces.Box(np.concatenate((spaces.object_position.gym.low, spaces.object_orientation.gym.low)),
                                            np.concatenate((spaces.object_position.gym.high, spaces.object_orientation.gym.high)))
        observation_space = gym.spaces.Box(
            np.concatenate((spaces.robot_position.gym.low, spaces.robot_velocity.gym.low, spaces.robot_torque.gym.low)),
            np.concatenate((spaces.robot_position.gym.high, spaces.robot_velocity.gym.high, spaces.robot_torque.gym.high))
        )

        self.observation_space = gym.spaces.Dict(
            {
                "observation": observation_space,
                "desired_goal": object_state_space,
                "achieved_goal": object_state_space,
            }
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal : Current pose of the object.
            desired_goal : Goal pose of the object.
            info : An info dictionary containing a field "difficulty"
                which specifies the difficulty level.

        Returns:
            float: The reward that corresponds to the provided achieved goal
            w.r.t. to the desired goal. Note that the following should always
            hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """

        if self.sparse_reward:
            return - np.float32((move_cube.evaluate_state(
                                move_cube.Pose(desired_goal[0:3], desired_goal[3:7]),
                                move_cube.Pose(achieved_goal[0:3], achieved_goal[3:7]),
                                1,
                            ) > 0.01))
        else:
            return move_cube.evaluate_state(
                                move_cube.Pose.from_dict(desired_goal),
                                move_cube.Pose.from_dict(achieved_goal),
                                1,
                            )

    def goal_observation(self, observation):


        goal_observation = {
            "observation": np.concatenate((observation['observation']['position'],
                                           observation['observation']['velocity'],
                                           observation['observation']['torque'])),
            "desired_goal": np.concatenate((self.goal['position'], self.goal['orientation'])),
            "achieved_goal": np.concatenate((observation['achieved_goal']['position'], observation['achieved_goal']['orientation']))
            }

        return goal_observation

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        ``reset()`` to reset this environment's state.

        Args:
            action: An action provided by the agent (depends on the selected
                :class:`ActionType`).

        Returns:
            tuple:

            - observation (dict): agent's observation of the current
              environment.
            - reward (float) : amount of reward returned after previous action.
            - done (bool): whether the episode has ended, in which case further
              step() calls will return undefined results.
            - info (dict): info dictionary containing the difficulty level of
              the goal.
        """
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # Use observations of step t + 1 to follow what would be expected
            # in a typical gym environment.  Note that on the real robot, this
            # will not be possible
            observation = self.goal_observation(self._create_observation(t + 1))

            reward = self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

        is_done = self.step_count == move_cube.episode_length

        return observation, reward, is_done, self.info

    def reset(self):
        # reset simulation
        del self.platform

        # initialize simulation
        initial_robot_position = (
            TriFingerPlatform.spaces.robot_position.default
        )
        initial_object_pose = self.initializer.get_initial_state()
        goal_object_pose = self.initializer.get_goal()

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.goal = {
            "position": goal_object_pose.position,
            "orientation": goal_object_pose.orientation,
        }

        # visualize the goal
        if self.visualization:
            self.goal_marker = visual_objects.CubeMarker(
                width=0.065,
                position=goal_object_pose.position,
                orientation=goal_object_pose.orientation,
                physicsClientId=self.platform.simfinger._pybullet_client_id,
            )

        self.info = {"difficulty": self.initializer.difficulty}

        self.step_count = 0

        return self.goal_observation(self._create_observation(0))
