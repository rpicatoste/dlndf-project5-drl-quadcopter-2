import numpy as np
from physics_sim import PhysicsSim
from collections import defaultdict

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self,
                 init_pose = None,
                 init_velocities = None,
                 init_angle_velocities = None,
                 runtime = 5.,
                 target_pos = None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6

        self.action_low = 50
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 


    def distance_reward(self):
        current_position = self.sim.pose[:3]
        target_position = self.target_pos
        #reward = -1.0 * (abs(current_position - target_position)).sum()
        # Since the norm is positive, tanh will give a value between 0 and 1.
        reward = -4.0 * np.tanh(np.linalg.norm(current_position - target_position) * 0.10)

        return reward

    def angles_reward(self):
        # Penalize only theta, which may make the quadrucopter roll upside-down.
        # Since we do the abs, it is always positive, tanh will give a value between 0 and 1.
        # reward = -1.0 * np.tanh(np.abs(self.sim.pose[4]) * 0.5)
        reward = -1.0 * np.tanh(self.sim.pose[4]**2 * 0.5)
        return reward


    def angular_speed_reward(self):
        # Penalizing angular speed will favor more stable motion.
        reward = -0.1 * np.tanh(np.abs(self.sim.angular_v).sum() * 0.5)
        return reward

    def similar_rotors_reward(self, rotor_speeds):
        # Penalize if rotor actions are too different.
        reward = 0
        avg_rotor = np.mean(rotor_speeds)
        mean_diff = np.mean([np.abs(avg_rotor - rotor_speed) for rotor_speed in rotor_speeds])
        minimum_considered = 0
        if mean_diff > minimum_considered:

            reward = -2.0*sigmoid((mean_diff-minimum_considered)*10/(900-minimum_considered)-5)

        return reward

    def get_reward(self, rotor_speeds):
        """Uses current pose of sim to return reward."""
        rewards = defaultdict(float)

        rewards['surviving'] = 1.0 if self.sim.done == False else -10.0
        rewards['distance'] = self.distance_reward()
        rewards['angles'] = self.angles_reward()
        rewards['angular_speed'] = self.angles_reward()
        rewards['similar_rotors'] = self.similar_rotors_reward(rotor_speeds)

        #rewards['test'] = -sum([np.abs(x-505) for x in rotor_speeds])

        reward = sum([x for x in rewards.values()])
        return reward, rewards

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        rewards = defaultdict(float)
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            instant_reward, new_rewards = self.get_reward(rotor_speeds)
            reward += instant_reward
            for key, value in new_rewards.items():
                rewards[key] += value
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        # if self.sim.reached_limits:
        #     reward -= 50

        return next_state, reward, done, rewards

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state