import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agents.striker.striker import StrikerAgent
from reinforcement_learning.model import QNetwork
from reinforcement_learning.dqn_agent import DQNCoreAgent as DQNAgent  # asumsi ada kelas pembungkus

class DQNStriker(StrikerAgent):
    def __init__(self, team="A", seed=0, state_size=9, action_space_size=7):
        super().__init__(team=team)  # perbaikan: sekarang mewarisi StrikerAgent
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.state_size = state_size
        self.action_space_size = action_space_size
        # inisialisasi DQN (sesuaikan dengan konstruktor DQNAgent Anda)
        self.dqn = DQNAgent(state_size=self.state_size, action_size=self.action_space_size, seed=seed)
        self.last_state_vec = None
        self.last_action_idx = None

    def get_state(self, snapshot):
        return snapshot

    def state_to_vector(self, state):
        players = state.get('players', [])
        me = players[0] if players else {}
        ball = state.get('ball', {})
        px = me.get('x', 0.0)
        py = me.get('y', 0.0)
        bx = ball.get('x', 50.0)
        by = ball.get('y', 37.5)
        bvx = ball.get('vx', 0.0)
        bvy = ball.get('vy', 0.0)
        dx = bx - px
        dy = by - py
        dist_ball = (dx*dx + dy*dy) ** 0.5
        ball_speed = (bvx*bvx + bvy*bvy) ** 0.5
        has_ball = 1.0 if state.get('ball_controller', -1) == 0 else 0.0
        vec = np.array([
            px/100.0,
            py/75.0,
            bx/100.0,
            by/75.0,
            dx/100.0,
            dy/75.0,
            dist_ball/125.0,
            has_ball,
            ball_speed/30.0
        ], dtype=np.float32)
        return vec

    def select_action(self, state):
        vec = self.state_to_vector(state)
        self.last_state_vec = vec
        if hasattr(self.dqn, 'act'):
            action_idx = self.dqn.act(vec)
        elif hasattr(self.dqn, 'choose_action'):
            action_idx = self.dqn.choose_action(vec)
        else:
            action_idx = np.random.randint(0, self.action_space_size)
        self.last_action_idx = action_idx
        return action_idx

    def action_index_to_dict(self, idx):
        px = self.last_state_vec[0] * 100.0
        py = self.last_state_vec[1] * 75.0
        bx = self.last_state_vec[2] * 100.0
        by = self.last_state_vec[3] * 75.0
        has_ball = self.last_state_vec[7] >= 0.5
        if idx == 0:
            return {'type': 'move', 'target': (px, py), 'speed': 0.0}
        if idx == 1:
            return {'type': 'move', 'target': (bx, by), 'speed': 6.0}
        if idx == 2:
            return {'type': 'control'}
        if idx == 3:
            return {'type': 'pass', 'target': (50.0, 37.5), 'power': 10.0}
        if idx == 4:
            return {'type': 'shoot', 'power': 12.0}
        if idx == 5:
            tx = px + (bx - px) * 0.5
            ty = py + (by - py) * 0.5
            return {'type': 'move', 'target': (tx, ty), 'speed': 5.0}
        if idx == 6:
            if has_ball:
                return {'type': 'move', 'target': (px, py), 'speed': 0.0}
            return {'type': 'move', 'target': (bx, by), 'speed': 4.0}
        return {'type': 'move', 'target': (px, py), 'speed': 0.0}

    def learn(self, reward, new_state, done):
        if self.last_state_vec is None or self.last_action_idx is None:
            return
        next_vec = self.state_to_vector(new_state)
        if hasattr(self.dqn, 'step'):
            self.dqn.step(self.last_state_vec, self.last_action_idx, reward, next_vec, done)
        elif hasattr(self.dqn, 'learn'):
            self.dqn.learn(self.last_state_vec, self.last_action_idx, reward, next_vec, done)
        self.last_state_vec = next_vec
        if done:
            self.last_action_idx = None
