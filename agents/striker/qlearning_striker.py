import math
import random
import numpy as np
from numpy.ma.core import indices
from pygame.event import set_keyboard_grab

from agents.base_agent import FootballAgent
from agents.striker.striker import StrikerAgent

#define the discrete action your strike can take
ACTIONS = ['control', 'move_to_ball', 'shoot', 'pass', 'run_with_ball', 'tackle', 'block', 'off_ball_run', 'idle']

class QLearningStriker(StrikerAgent):
    def __init__(self, team,
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 epsilon: float = 0.2):
        super().__init__(team)
        self.alpha = alpha  #learning rate
        self.gamma = gamma  #discount factor
        self.epsilon = epsilon #exploration rate
        self.q_table = {}   #maps state tuples --> list of Q-Values
        self.last_state = None
        self.last_action = None
        self.last_action_dict = None

    def discretize(self, state: dict):
        #example discretization of your StrikerAgent state
        dist_to_ball = min(int(state['dist_to_ball'] // 20), 4)
        dist_to_goal = min(int(state['dist_to_goal'] // 20), 4)
        dist_to_nearest_opponent = min(int(state['nearest_opponents_dist'] // 10), 4)
        best_teammate_pos = getattr(state, 'best_teammate_pos', state['self_pos'])
        dist_to_best_teammate = min(int(math.hypot(best_teammate_pos[0] - state['self_pos'][0], best_teammate_pos[1] - state['self_pos'][1]) // 10), 4)
        # has_ball = int(state['has_ball'])
        # zone: 0=defense, 1=midfield, 2=attack
        self_pos_x = state['self_pos'][0]
        zone_vertical = 0 if self_pos_x < 30 else (2 if self_pos_x > 70 else 1)
        if state['ball_controller'] is None:
            ball_controller = 0
        elif state['ball_controller'] == 'me':
            ball_controller = 1
        elif state['ball_controller'] == 'teammate':
            ball_controller = 2
        else:
            ball_controller = 3
        return (dist_to_goal, dist_to_ball, dist_to_nearest_opponent, zone_vertical, ball_controller, dist_to_best_teammate)

    def decide_action(self, state: dict, ball_controller=None):
        key =  self.discretize(state)
        #initiaaalize unseen state torh zero Qs
        if key not in self.q_table:
            self.q_table[key] = [0.0] * len(ACTIONS)
        self.ball_controller = ball_controller

        # epsilon-greedy but must has ball controlled
        if state['ball_controller'] is None:
            if random.random() < self.epsilon:
                if state['dist_to_ball'] < self.control_range: #1 meter
                    idx = random.choice([0, 1, 6, 7, 8]) #only control, move_to_ball, tackle, block, off_ball_run, idle
                else:
                    idx = random.choice([1, 6, 7, 8]) #only move_to_ball, block, off_ball_run, idle
            else:
                if state['dist_to_ball'] < self.control_range:
                    temp_arr = self.q_table[key]
                    indices = [0, 1, 6, 7, 8]
                    subset = [temp_arr[i] for i in indices]
                    idx_in_subset = int(np.argmax(subset))
                    idx = indices[idx_in_subset]
                else:
                    temp_arr = self.q_table[key]
                    indices = [1, 6, 7, 8]
                    subset = [temp_arr[i] for i in indices]
                    idx_in_subset = int(np.argmax(subset))
                    idx = indices[idx_in_subset]
        elif state['ball_controller'] == 'me':
            if random.random() < self.epsilon:
                idx = random.choice([2, 3, 4, 8]) #only shoot, pass, run_with_ball, idle
            else:
                temp_arr = self.q_table[key]
                indices = [2, 3, 4, 8]
                subset = [temp_arr[i] for i in indices]
                idx_in_subset = int(np.argmax(subset))
                idx = indices[idx_in_subset]
        elif state['ball_controller'] == 'teammate':
            if random.random() < self.epsilon:
                idx = random.choice([1, 7, 8]) #only move_to_ball, off_ball_run, idle
            else:
                temp_arr = self.q_table[key]
                indices = [1, 7, 8]
                subset = [temp_arr[i] for i in indices]
                idx_in_subset = int(np.argmax(subset))
                idx = indices[idx_in_subset]
        else:
            #if ball controlled by opponent
            if random.random() < self.epsilon:
                if state['dist_to_ball'] < self.tackle_range:
                    idx = random.choice([1, 5, 6, 7, 8]) #only move_to_ball, tackle, block, off_ball_run, idle
                else:
                    idx = random.choice([1, 6, 7, 8]) #only move_to_ball, block, off_ball_run, idle
            else:
                if state['dist_to_ball'] < self.tackle_range:
                    temp_arr = self.q_table[key]
                    indices = [1, 5, 6, 7, 8] #only move_to_ball, tackle, block, off_ball_run, idle
                    subset = [temp_arr[i] for i in indices]
                    idx_in_subset = int(np.argmax(subset))
                    idx = indices[idx_in_subset]
                else:
                    temp_arr = self.q_table[key]
                    indices = [1, 6, 7, 8]
                    subset = [temp_arr[i] for i in indices]
                    idx_in_subset = int(np.argmax(subset))
                    idx = indices[idx_in_subset]

        self.last_state = key
        self.last_action = idx

        #map discrete action back to your action dict
        action_name = ACTIONS[idx]
        if action_name == "shoot":
            self.last_action_dict = {'type': 'shoot', 'target': state['goal_pos']}
            return self.last_action_dict
        if action_name == "pass":
            self.last_action_dict = {'type': 'pass', 'target': state['best_teammate_pos']}
            return self.last_action_dict
        if action_name == "off_ball_run" and state['dist_to_goal'] < 25:
            self.last_action_dict = {'type': 'move', 'target': (
                            state['ball_pos'][0] + (random.randint(-10, 10)),
                            state['ball_pos'][1] + (random.randint(-10, 10)),
                        )}
            return self.last_action_dict
        elif action_name == "off_ball_run":
            self.last_action_dict = {'type': 'move', 'target': (
                            state['ball_pos'][0] + (random.randint(-20, 20)),
                            state['ball_pos'][1] + (random.randint(-30, 30)),
                        )}
            return self.last_action_dict
        if action_name == "control":
            self.last_action_dict = {'type': 'control', 'target': state['ball_pos']}
            return self.last_action_dict
        if action_name == "run_with_ball":
            self.last_action_dict = {'type': 'run_with_ball', 'target': (
                        state['self_pos'][0] + (10 if self.team.upper() == "A" else -10),
                        state['self_pos'][1] + (random.randint(-5, 5)),
                    )}
            return self.last_action_dict
        if action_name == "move_to_ball":
            self.last_action_dict = {'type': 'move', 'target': state['ball_pos']}
            return self.last_action_dict
        if action_name == "tackle":
            self.last_action_dict = {'type': 'tackle', 'target': state['ball_pos']}
            return self.last_action_dict
        if action_name == "block":
            self.last_action_dict = {'type': 'block', 'target': state['ball_pos']}
            return self.last_action_dict
        #default move_to_ball
        self.last_action_dict = {'type': 'move', 'target': state['self_pos']}
        return self.last_action_dict

    def learn(self, reward: float, new_state: dict):
        """ Q(s, a) <-- Q(s, a) + alpha [r + gamma max_a' Q(s', a') - Q(s, a)] """
        new_key = self.discretize(new_state)
        #init new state Qs if necessary
        if new_key not in self.q_table:
            self.q_table[new_key] = [0.0] * len(ACTIONS)

        old_q = self.q_table[self.last_state][self.last_action]
        future_q = max(self.q_table[new_key])
        #update rule
        self.q_table[self.last_state][self.last_action] = (
            old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        )