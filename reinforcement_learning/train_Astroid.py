import gymnasium as gym
import time
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from dqn_agent import Agent

EXPERIMENT_NAME = "Astroids_Tanh"
SCORE_GOAL      = 10_000
MAXIMUM_EPISODE = 5_000

def normalise_values(values : np.ndarray) -> np.ndarray:
    return (values - values.min()) / np.ptp(values)

def dqn(
        env, 
        agent : Agent, 
        n_episodes  = MAXIMUM_EPISODE, 
        max_t       = 1000, 
        eps_start   = 1.0, 
        eps_end     = 0.01, 
        eps_decay   = 0.995, 
        goal_score  = SCORE_GOAL
    ) -> tuple[list[float], list[float]]:
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    loss   = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    current_best = 0
    for i_episode in range(1, n_episodes+1):
        state, info = env.reset()
        state = normalise_values(state)

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, info, _ = env.step(action)
            next_state = normalise_values(next_state)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break 
                
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        loss.append(agent.latest_loss())  # save most recent loss

        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

        if np.mean(scores_window) >= current_best:
            torch.save(agent.qnetwork_local.state_dict(), f'{EXPERIMENT_NAME}_best.pth')
            current_best = np.mean(scores_window)
        
        if np.mean(scores_window) >= goal_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), f'{EXPERIMENT_NAME}_final.pth')
            break
            
    return scores, loss

def running_average(lst : list) -> list:
    cumsum = np.cumsum(lst)
    return [cumsum[i] / (i + 1) for i in range(len(cumsum))]


if __name__ == "__main__":
    env   = gym.make('ALE/Asteroids-ram-v5')
    agent = Agent(state_size = 128, action_size = 5, seed = 42)

    scores, loss = dqn(env, agent)

    rav_scores = running_average(scores)
    rav_loss   = running_average(loss)

    df = pd.DataFrame({
        f'scores_{EXPERIMENT_NAME}': scores,
        f'loss_{EXPERIMENT_NAME}'  : loss
    })
    df.to_csv(f'./{EXPERIMENT_NAME}.csv', index=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)),     scores,           label = 'Scores')
    plt.plot(np.arange(len(rav_scores)), rav_scores, '-.', label = "Averaged Score")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(f'./{EXPERIMENT_NAME}_plot.jpg')

    from model import QNetwork
    m = QNetwork(128, 5, 0)
    m.load_state_dict(torch.load(f"{EXPERIMENT_NAME}_best.pth", map_location = torch.device('cpu')))

    with torch.no_grad():
        example_forward_input = torch.rand(1, 128)
        module = torch.jit.trace(m, example_forward_input)
    torch.jit.save(module, f"{EXPERIMENT_NAME}.jit.pth")

    print("Complete !")



