import tensorflow as tf
import tensorflow_quantum as tfq
import wandb
import gym, cirq, sympy
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
tf.get_logger().setLevel('ERROR')
import time
from cirq_helper import *
from qlearning_helper import *
from qlearning_helper import *
from pqc_helper import *

PATH = "/content/drive/MyDrive/quantum_rl/lunar_checkpoint.h5"

def interact_env(state, model, epsilon, n_actions, env):
    # Preprocess state
    # print("state type : ", type(state))
    # print("state : ", state[0])

    if isinstance(state, tuple):
      state = state[0]

    state_array = np.array(state) 
    state = tf.convert_to_tensor([state_array])

    # Sample action
    coin = np.random.random()
    if coin > epsilon:
        q_vals = model([state])
        action = int(tf.argmax(q_vals[0]).numpy())
    else:
        action = np.random.choice(n_actions)

    # Apply sampled action in the environment, receive reward and next state
    next_state, reward, done, _, _ = env.step(action)

    interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(),
                   'reward': reward, 'done':np.float32(done)}

    return interaction

if __name__ == "__main__":
  wandb.login(key='')

  wandb.init(
        # set the wandb project where this run will be logged
        project="quantum_rl_lunar",
        name="base"
    )

  env = gym.make("Acrobot-v1")
  
  n_qubits = 6# Dimension of the state vectors in acrobot
  n_layers = 3 # Number of layers in the PQC
  # n_layers = 1 # Number of layers in the PQC
  n_actions = 3 # Number of actions in acrobot

  qubits = cirq.GridQubit.rect(1, n_qubits)
  ops = [cirq.Z(q) for q in qubits]
  # observables = [ops[0]*ops[1], ops[2]*ops[3]] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1
  observables = [reduce((lambda x, y: x * y), ops)]


  model = generate_model_Qlearning(qubits, n_layers, n_actions, observables, False) 
  model_target = generate_model_Qlearning(qubits, n_layers, n_actions, observables, True)

  model_target.set_weights(model.get_weights())



  gamma = 0.99
  n_episodes = 2000

  # Define replay memory
  max_memory_length = 10000 # Maximum replay length
  replay_memory = deque(maxlen=max_memory_length)

  epsilon = 1.0  # Epsilon greedy parameter
  epsilon_min = 0.01  # Minimum epsilon greedy parameter
  decay_epsilon = 0.99 # Decay rate of epsilon greedy parameter
  batch_size = 16
  steps_per_update = 10 # Train the model every x steps
  steps_per_target_update = 30 # Update the target model every x steps

  optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
  optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
  optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

  # Assign the model parameters to each optimizer
  w_in, w_var, w_out = 1, 0, 2

 

  episode_reward_history = []
  step_count = 0
  
  replay_memory = []
  for episode in range(n_episodes):
      start_time = time.time()
      episode_reward = 0
      state = env.reset()

      while True:
          
          # Interact with env
          interaction = interact_env(state, model, epsilon, n_actions, env)
          
          # Store interaction in the replay memory
          replay_memory.append(interaction)

          state = interaction['next_state']
          
          episode_reward += interaction['reward']
          step_count += 1

          # Update model
          if step_count % steps_per_update == 0:
              # Sample a batch of interactions and update Q_function
              training_batch = np.random.choice(replay_memory, size=batch_size)
              
              Q_learning_update(model_target, np.asarray([x['state'] for x in training_batch]),
                                np.asarray([x['action'] for x in training_batch]),
                                np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                np.asarray([x['next_state'] for x in training_batch]),
                                np.asarray([x['done'] for x in training_batch], dtype=np.float32),
                                model, gamma, n_actions, optimizer_in, optimizer_var, optimizer_out,
                                w_in, w_var, w_out)

          # Update target model
          if step_count % steps_per_target_update == 0:
              model_target.set_weights(model.get_weights())

          # Check if the episode is finished
          if interaction['done']:
              break

      # Decay epsilon
      epsilon = max(epsilon * decay_epsilon, epsilon_min)
      episode_reward_history.append(episode_reward)
      if (episode+1)%10 == 0:
          avg_rewards = np.mean(episode_reward_history[-10:])
          print("Episode {}/{}, average last 10 rewards {}".format(
              episode+1, n_episodes, avg_rewards))
              
          wandb.log({"rewards ": avg_rewards})
          # torch.save(model.state_dict(), PATH)
          print("model : ", model)
          # model.save(PATH)
          model.save_weights("acrobot_checkpoint")
          print("time : ", time.time()-start_time)
          
          if avg_rewards >= 500.0:
              break

  plt.figure(figsize=(10,5))
  plt.plot(episode_reward_history)
  plt.xlabel('Epsiode')
  plt.ylabel('Collected rewards')
  plt.show()
