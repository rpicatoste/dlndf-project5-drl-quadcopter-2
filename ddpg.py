
import numpy as np
import tensorflow as tf
import json

from task import Task
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

OU = OU()       #Ornstein-Uhlenbeck Process

def playGame(train_indicator=0):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    MU = 200.0
    THETA = 1.0
    SIGMA = 10.0

    action_dim = 4
    action_repeat = 3
    state_dim = action_repeat*6 #action_repeat * pose

    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # Generate a Torcs environment
    #env = TorcsEnv(vision=vision, throttle=True,gear_change=False)
    target_pos = np.array([0.0, 0.0, 10.0])
    init_pose = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    init_velocities = np.array([0.0, 0.0, 0.0])
    task = Task(init_pose=init_pose,
                init_velocities=init_velocities,
                target_pos=target_pos)

    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights(r"models\actormodel.h5")
        critic.model.load_weights(r"models\criticmodel.h5")
        actor.target_model.load_weights(r"models\actormodel.h5")
        critic.target_model.load_weights(r"models\criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i_episode in range(episode_count):

        print("Episode : " + str(i_episode) + " Replay Buffer " + str(buff.count()))

        # RESET
        #ob = env.reset()
        task.sim.reset()
        s_t = np.concatenate([task.sim.pose] * task.action_repeat)

        #s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.
        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1,action_dim])
            noise_t = np.zeros([1,action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            if train_indicator:
                noise_t[0][0] = max(epsilon, 0) * OU.function(a_t_original[0][0], MU, THETA, SIGMA)
                noise_t[0][1] = max(epsilon, 0) * OU.function(a_t_original[0][1], MU, THETA, SIGMA)
                noise_t[0][2] = max(epsilon, 0) * OU.function(a_t_original[0][2], MU, THETA, SIGMA)
                noise_t[0][3] = max(epsilon, 0) * OU.function(a_t_original[0][3], MU, THETA, SIGMA)
            else:
                noise_t[0][0] = 0
                noise_t[0][1] = 0
                noise_t[0][2] = 0
                noise_t[0][3] = 0

            #The following code do the stochastic brake
            #if random.random() <= 0.1:
            #    print("********Now we apply the brake***********")
            #    noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
            a_t[0][3] = a_t_original[0][3] + noise_t[0][3]

            s_t1, r_t, done, rewards_dict = task.step(a_t[0])

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            #Do the batch update
            batch      = buff.getBatch(BATCH_SIZE)
            states     = np.asarray([e[0] for e in batch])
            actions    = np.asarray([e[1] for e in batch])
            rewards    = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones      = np.asarray([e[4] for e in batch])
            y_t        = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict(
                [new_states, actor.target_model.predict(new_states)]
            )

            #y_t = rewards + (1 - dones) * GAMMA * target_q_values
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            print("Episode {: 3}. Step {}. Reward {:8.2f}. Loss {:10.0f}. Action({:6.1f},{:6.1f},{:6.1f},{:6.1f}).  Noise({:6.1f},{:6.1f},{:6.1f},{:6.1f}).".format(
                i_episode,
                step,
                r_t,
                loss,
                *[rotor for rotor in a_t_original[0]],
                *[rotor for rotor in noise_t[0]])
            )

            step += 1
            if done:
                break

        if np.mod(i_episode, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights(r"models\actormodel.h5", overwrite=True)
                with open(r"models\actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights(r"models\criticmodel.h5", overwrite=True)
                with open(r"models\criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i_episode) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")


    print("Finish.")

if __name__ == "__main__":
    playGame(True)
