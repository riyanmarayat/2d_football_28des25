import numpy as np
from agents.striker.striker import StrikerAgent
from reinforcement_learning.dqn_agent import DQNCoreAgent

ACTIONS = ['control', 'move_to_ball', 'shoot', 'pass',
           'run_with_ball', 'tackle', 'block', 'off_ball_run', 'idle']

class DQNStriker(StrikerAgent):
    def __init__(self, team, seed=0):
        super().__init__(team)
        #TODO sementara state_size diisi angka, nanti disesuaikan
        self.dqn = DQNCoreAgent(state_size=9, action_size=len(ACTIONS), seed=seed)
        self.last_state_vec = None
        self.last_action_idx = None

    def encode_state(self, s):
        return np.array([
            s['self_pos'][0], s['self_pos'][1],
            s['ball_pos'][0], s['ball_pos'][1],
            s['goal_pos'][0], s['goal_pos'][1],
            s['dist_to_ball'],
            s['dist_to_goal'],
            s['nearest_opponents_dist'],
        ], dtype=np.float32)

    def decide_action(self, state, ball_controller=None, eps=0.1):
        self.ball_controller = ball_controller
        s_vec = self.encode_state(state)
        a_idx = self.dqn.act(s_vec, eps)   # pakai DQN untuk pilih aksi
        self.last_state_vec = s_vec
        self.last_action_idx = a_idx

        #TODO mapping index → action dict (bisa copy dari QLearningStriker)
        action_name = ACTIONS[a_idx]
        # contoh beberapa:
        if action_name == "shoot":
            return {'type': 'shoot', 'target': state['goal_pos']}
        if action_name == "pass":
            return {'type': 'pass', 'target': state['best_teammate_pos']}
        if action_name == "move_to_ball":
            return {'type': 'move', 'target': state['ball_pos']}
        # dst, sama seperti di QLearningStriker
        return {'type': 'move', 'target': state['self_pos']}

    def learn(self, reward, next_state, done=False):
        next_vec = self.encode_state(next_state)
        self.dqn.step(self.last_state_vec, self.last_action_idx, reward, next_vec, done)

    def action_index_to_dict(self, idx):
        # Sesuaikan dengan definisi aksi Anda
        # Contoh mapping sederhana:
        mapping = {
            0: {'type': 'move', 'target': (50.0, 37.5), 'speed': 6.0},
            1: {'type': 'control'},
            2: {'type': 'pass', 'target': (70.0, 37.5), 'power': 10.0},
            3: {'type': 'shoot', 'power': 12.0},
        }
        return mapping.get(idx, {'type': 'noop', 'idx': idx})
