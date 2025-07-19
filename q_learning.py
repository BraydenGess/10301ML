import numpy as np
from environment import MountainCar, GridWorld
import random
import sys

def initialize_environment(env,mode):
    if env == 'mc':
        return MountainCar(mode=mode)
    return GridWorld(mode=mode)

def get_vector(state,env):
    s = np.zeros(env.state_space)
    for key in state:
        s[key] = state[key]
    return s

def get_action(state,env,weights,bias):
    s = get_vector(state,env)
    action_vector = np.matmul(s,weights)
    action_vector = action_vector+bias
    return action_vector,np.argmax(action_vector)

def q_learning(env,mode,episodes,max_iterations,epsilon,gamma,learning_rate):
    weights = np.zeros((env.state_space, env.action_space))
    bias = 0
    episode_rewards = np.zeros(episodes)
    for episode in range(episodes):
        state = env.reset()
        reward = 0
        goal = False
        counter = 0
        while (counter < max_iterations and goal==False):
            probability = random.random()
            action_vector,ACTION = get_action(state,env,weights,bias)
            if probability <= epsilon:
                ACTION = random.randint(0,env.action_space-1)
            (next_state,reward_gathered,goal) = env.step(ACTION)
            reward += reward_gathered
            ###Update weights and bias
            current_state_vector = get_vector(state,env)
            next_state_action_vector,NULL_ACTION = get_action(next_state,env,weights,bias)
            ymaxqsaw = (gamma * np.max(next_state_action_vector))
            rplusymaxqsaw  = reward_gathered + ymaxqsaw
            middle_piece = action_vector[ACTION] - rplusymaxqsaw
            bias_update = learning_rate * middle_piece
            weights_update = bias_update * current_state_vector
            weights[:, ACTION] = weights[:, ACTION] - weights_update
            bias = bias - bias_update
            ###Advance to next iteration
            state = next_state
            counter += 1
        episode_rewards[episode] = reward
    return bias,weights,episode_rewards

def write_to_files(weight_out_file,bias,weights,returns_out,episode_rewards):
    with open(weight_out_file,'w') as f:
        f.write(str(bias))
        f.write('\n')
        rows,cols = weights.shape
        for row in range(rows):
            for col in range(cols):
                f.write(str(weights[row][col]))
                if row != (rows-1) or col != (cols-1):
                    f.write('\n')
    f.close()
    with open(returns_out, 'w') as f:
        (rows,) = episode_rewards.shape
        for i in range(rows):
            f.write(str(episode_rewards[i]))
            if (i != rows-1):
                f.write('\n')
    f.close()

def main():
    mode = sys.argv[2]
    env = initialize_environment(sys.argv[1],mode)
    weight_out = sys.argv[3]
    returns_out = sys.argv[4]
    episodes = int(sys.argv[5])
    max_iterations = int(sys.argv[6])
    epsilon = float(sys.argv[7])
    gamma = float(sys.argv[8])
    learning_rate = float(sys.argv[9])
    bias,weights,episode_rewards = q_learning(env, mode, episodes, max_iterations, epsilon, gamma, learning_rate)
    write_to_files(weight_out, bias, weights,returns_out,episode_rewards)

main()
