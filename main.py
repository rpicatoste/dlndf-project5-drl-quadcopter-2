#%%
import time
import sys
import os
import csv
import numpy as np
from task import Task
from agents.agent import DDPG
from plot_functions import plot_results, plot_training_historic
from collections import defaultdict
import copy

import matplotlib.pyplot as plt
from agents.ou_noise import OUNoise

## Modify the values below to give the quadcopter a different starting position.
file_output = 'data.txt'                         # file name for saved results
plt.close('all')


# Run task with agent
def run_test_episode(agent : DDPG, task : Task, file_output):
    print('\nRunning test episode ...')

    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4','reward']
    results = {x : [] for x in labels}

    aux_noise = copy.copy(agent.noise)
    agent.noise = OUNoise(agent.action_size, 0.0, 0.0, 0.0)
    
    state = agent.reset_episode() # start a new episode
    rewards_lists = defaultdict(list)
    print('state', state)
    print('state.shape', state.shape)
    
    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            rotor_speeds = agent.act(state)
            #rotor_speeds = [405]*4
            #rotor_speeds = [800, 10, 10, 10]
            next_state, reward, done, new_rewards = task.step(rotor_speeds)
            for key, value in new_rewards.items():
                rewards_lists[key].append(value)

            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds) + [reward]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)

            state = next_state
            if done:
                break

     # Restore noise
    agent.noise = copy.copy(aux_noise)

    print('Finished test episode!\n')
    return results, rewards_lists

#%% Parameters
exploration_mu = 200
exploration_theta = 1.0#0.15
exploration_sigma = 100.0#0.2
buffer_size = 100000
batch_size = 64
gamma = 0.99
tau = 0.001
actor_learning_rate = 0.0001
critic_learning_rate = 0.001

num_episodes = 1000 # 1000

#%% Training with agen
print('\n\nStart training...')
num_episodes_to_plot = max(100, num_episodes/5)
target_pos      = np.array([ 0.0, 0.0, 10.0])
init_pose       = np.array([ 0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
init_velocities = np.array([ 0.0, 0.0,  0.0])
task = Task(init_pose = init_pose,
           init_velocities = init_velocities,
           target_pos=target_pos)
agent = DDPG(task,
             exploration_mu =exploration_mu,
             exploration_theta = exploration_theta,
             exploration_sigma = exploration_sigma,
             buffer_size = buffer_size,
             batch_size = batch_size,
             gamma = gamma,
             tau = tau,
             actor_learning_rate = actor_learning_rate,
             critic_learning_rate = critic_learning_rate
             )

results, rewards_lists = run_test_episode(agent, task, file_output)
plot_results(results, target_pos, 'Run without training', rewards_lists)

#plt.show();import sys;sys.exit()
# Train
max_reward = -np.inf
last_i_max_reward = 300
history = {'total_reward' : [], 'score' : [], 'i_episode' : []}
start = time.time()
done = False
i_episode = 1
stuck_counter = 0
while i_episode < num_episodes+1:
    state = agent.reset_episode() # start a new episode
    t_episode = 0
    time_step_episode = 0
    cum_sum_actions = np.array([0.0]*4)

    while True:
        action = agent.act(state, i_episode%1000 == 0)
        next_state, reward, done, _ = task.step(action)

        agent.step(action, reward, next_state, done)
        state = next_state

        cum_sum_actions += action
        time_step_episode += 1

        t_episode += 0.06 # each step is 3 times 20 ms (50Hz)
        if done:
            if t_episode > 0.1 + i_episode*0.001:

                i_episode += 1

                # Slowly decrease noise if everything goes all right.
                agent.noise.sigma = exploration_sigma / i_episode
                agent.noise.theta = exploration_theta / i_episode
                #agent.noise.theta *= 0.9975

                if len(history['i_episode'])>1:
                    history['i_episode'].append(history['i_episode'][-1] + 1)
                else:
                    history['i_episode'].append(1)
                history['total_reward'].append(agent.total_reward)
                history['score'].append(agent.score)

                if stuck_counter > 1000:
                    stuck_counter -= 1000
                else:
                    stuck_counter = 0
            else:
                # do not count as episode if it didn't last a minimum.
                # Slowly increase noise to try new values.
                if agent.noise.sigma < exploration_sigma:
                    agent.noise.sigma = exploration_sigma / i_episode
                    agent.noise.theta = exploration_theta / i_episode
                    #agent.noise.sigma *= 1.001
                    #agent.noise.theta *= 1.001
                    #agent.noise.state *= 0.99

                stuck_counter += 1
                if stuck_counter > 9999:
                    stuck_counter = 0
                    agent.noise.reset()


            print("\rEpisode:{: 4d} (stuck:{: 5d}), score: {:7.1f}, reward: {:8.2f}, noise(sigma: {:6.3f}, theta: {:6.3f}, state, {:6.1f},{:6.1f},{:6.1f},{:6.1f})(action:{:6.1f},{:6.1f},{:6.1f},{:6.1f})".
                  format(i_episode,
                         stuck_counter,
                         agent.score,
                         agent.total_reward,
                         agent.noise.sigma,
                         agent.noise.theta,
                         *[rotor for rotor in agent.noise.state],
                         *[cum_sum_action/time_step_episode for cum_sum_action in cum_sum_actions]
                         ),
                  end="")

            break
    sys.stdout.flush()

    if i_episode%num_episodes_to_plot == 0 and stuck_counter == 0:
        results, rewards_lists = run_test_episode(agent, task, file_output)
        plot_results(results, target_pos, 'Run after training for {} episodes.'.format(i_episode), rewards_lists, i_episode)
    if (max_reward < reward) and (last_i_max_reward + 50 < i_episode):
        results, rewards_lists = run_test_episode(agent, task, file_output)
        plot_results(results, target_pos, 'New max at: {} episodes.'.format(i_episode), rewards_lists, i_episode)
        max_reward = reward
        last_i_max_reward = i_episode

print('\nTime training: {:.1f} seconds\n'.format(time.time() - start))

plot_training_historic(history)

# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print(task.sim.pose)
print(task.sim.v)
print(task.sim.angular_v)
    
results, rewards_lists = run_test_episode(agent, task, file_output)

plot_results(results, target_pos, 'Run after training for {} episodes.'.format(num_episodes), rewards_lists)

plt.show()


#%%
#plt.figure()
#plt.plot(results['time'], results['x_velocity'], label='x_hat')
#plt.plot(results['time'], results['y_velocity'], label='y_hat')
#plt.plot(results['time'], results['z_velocity'], label='z_hat')
#plt.legend()
#_ = plt.ylim()
#plt.show(block=False)
#
#
#plt.legend()
#_ = plt.ylim()
#plt.show(block=False)
#
#
#plt.figure()
#plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
#plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
#plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
#plt.legend()
#_ = plt.ylim()
#plt.show(block=False)




#%%

