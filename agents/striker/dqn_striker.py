import math
import random
from collections import deque
from typing import Dict, Tuple, Any, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([b[0] for b in batch], axis=0)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.stack([b[3] for b in batch], axis=0)
        dones = np.array([b[4] for b in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNStriker:
    """
    DQN untuk Striker dengan:
    - State feature konsisten (panjang 32) diekstrak dari snapshot simulator.
    - Action space lengkap dan stabil (13 aksi).
    - Epsilon-greedy, target network, dan experience replay.

    Action dict schema yang dihasilkan:
      {
        "move_vx": float,   # kecepatan X (unit/s)
        "move_vy": float,   # kecepatan Y (unit/s)
        "kick_power": float,# 0..1
        "kick_dir": (dx, dy)# unit vector arah kick, or None
      }
    Pastikan simulator.apply_action(player, action_dict, dt) mendukung field di atas.
    Jika simulator Anda menggunakan field lain, sesuaikan fungsi action_index_to_dict.
    """

    def __init__(
        self,
        team: str = "A",
        seed: Optional[int] = 42,
        state_dim: int = 48,
        n_actions: int = 13,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        min_buffer_to_learn: int = 1000,
        target_update_interval: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        max_move_speed: float = 12.0,   # unit per second
        sprint_multiplier: float = 1.6,
    ):
        self.team = team.upper()
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.min_buffer_to_learn = min_buffer_to_learn
        self.target_update_interval = target_update_interval
        self.train_steps = 0
        self.last_state: Optional[np.ndarray] = None

        # epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.max_move_speed = max_move_speed
        self.sprint_multiplier = sprint_multiplier

        if torch is None:
            raise ImportError("PyTorch is required for DQNStriker. Please install torch.")

        torch.manual_seed(seed or 0)

        self.policy_net = MLP(state_dim, n_actions)
        self.target_net = MLP(state_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # last_action_idx for training loop book-keeping
        self.last_action_idx: Optional[int] = None
        self._last_action_vel: Tuple[float, float] = (0.0, 0.0)
        self._last_action_kick: Tuple[float, Optional[Tuple[float, float]]] = (0.0, None)

        # Precompute action vectors
        self._actions = self._build_action_space()
        # cache indices untuk aksi gerak (vx,vy != 0 dan kick=0)
        self._move_indices = [
            i for i, a in enumerate(self._actions)
            if (abs(a.get("move_vx", 0.0)) > 1e-6 or abs(a.get("move_vy", 0.0)) > 1e-6)
            and abs(a.get("kick_power", 0.0)) < 1e-6
        ]

    # ------------- Public API -------------

    def select_action(self, snapshot: Dict[str, Any]) -> int:
        """
        menerima snapshot simulator, mengekstrak fitur state, lalu pilih idx aksi.
        """
        state_vec = self.extract_features(snapshot)
        if self.rng.random() < self.epsilon:
            # eksplorasi: 80% ke arah bola, 20% random
            if self.rng.random() < 0.8:
                action = self._action_towards_ball(snapshot)
            else:
                action = int(self.rng.integers(0, self.n_actions))
        else:
            with torch.no_grad():
                q = self.policy_net(torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())
        self.last_action_idx = action
        # cache velocity & kick so simulator can query via desired_velocity
        a = self._actions[action]
        self._last_action_vel = (float(a.get("move_vx", 0.0)), float(a.get("move_vy", 0.0)))
        self._last_action_kick = (float(a.get("kick_power", 0.0)), a.get("kick_dir", None))
        # update epsilon (linear decay)
        self._decay_epsilon()
        return action

    def desired_velocity(self, player, ball, field) -> Tuple[float, float]:
        """
        Dipanggil simulator untuk agen yang menyediakan desired_velocity.
        Mengembalikan v dari aksi terakhir agar tidak ditimpa logic default.
        """
        return self._last_action_vel

    def action_index_to_dict(self, action_idx: int) -> Dict[str, Any]:
        """
        Mengubah index aksi menjadi dict untuk simulator.
        """
        return self._actions[action_idx]

    def _action_towards_ball(self, snapshot: Dict[str, Any]) -> int:
        """Cari aksi gerak paling mendekati arah bola (heuristik eksplorasi)."""
        players: List[Dict[str, Any]] = snapshot.get("players", [])
        ball: Dict[str, Any] = snapshot.get("ball", {})
        if not players:
            return int(self.rng.integers(0, self.n_actions))
        sx, sy = float(players[0].get("x", 0.0)), float(players[0].get("y", 0.0))
        bx, by = float(ball.get("x", 0.0)), float(ball.get("y", 0.0))
        dx, dy = bx - sx, by - sy
        norm = math.sqrt(dx * dx + dy * dy)
        if norm > 1e-6:
            dx /= norm; dy /= norm
        best_idx = 0
        best_dot = -1e9
        for i in self._move_indices:
            a = self._actions[i]
            ax, ay = float(a.get("move_vx", 0.0)), float(a.get("move_vy", 0.0))
            amag = math.sqrt(ax*ax + ay*ay)
            if amag < 1e-6:
                continue
            ax /= amag; ay /= amag
            dot = ax * dx + ay * dy
            if dot > best_dot:
                best_dot = dot
                best_idx = i
        return best_idx

    def learn(self, reward: float, next_snapshot: Dict[str, Any], done: bool):
        """
        Menyimpan transition dan melakukan satu langkah update saat buffer cukup.
        """
        if self.last_state is None or self.last_action_idx is None:
            # Tidak ada state terakhir yang valid (misalnya di step pertama).
            return

        next_state_vec = self.extract_features(next_snapshot)
        self.buffer.push(self.last_state, self.last_action_idx, reward, next_state_vec, done)

        if len(self.buffer) < self.min_buffer_to_learn:
            self.last_state = next_state_vec
            return

        # Sample
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Q(s,a)
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # target Q
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)              # for argmax
            next_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)
            next_q_target = self.target_net(next_states_t).gather(1, next_actions)
            target = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # update last state
        self.last_state = next_state_vec

    # ------------- State Features -------------

    def extract_features(self, snapshot: Dict[str, Any]) -> np.ndarray:
        """
        Produces a fixed-length feature vector (size = self.state_dim).
        Assumes snapshot contains keys typically produced by your Simulator.snapshot().
        If your snapshot schema differs, adjust the extraction accordingly.

        Feature layout (48 total):
        - Self/ball kinematics: self pos/vel (4), ball pos/vel (4)
        - Tactical zones (mirrored for team B): self vertical lane one-hot (5), self horizontal third one-hot (3),
          ball vertical lane one-hot (5), ball horizontal third one-hot (3)
        - Possession: has_ball_self, has_ball_team_other, has_ball_opponent
        - Goal/relative vectors: opponent goal pos (2), self->ball dir+dist (3), ball->goal dir+dist (3)
        - Nearest entities: nearest teammate pos/vel (4), nearest opponent pos/vel (4)
        - Meta: team_side flag, ball_controller_self, stamina proxy, time_normalized, bias
        """
        field_info = snapshot.get("field", {})
        field_w = float(field_info.get("width", 100.0))
        field_h = float(field_info.get("height", 75.0))
        max_speed = self.max_move_speed * self.sprint_multiplier

        # players list of dicts:
        players: List[Dict[str, Any]] = snapshot.get("players", [])
        ball: Dict[str, Any] = snapshot.get("ball", {})
        controller = snapshot.get("ball_controller", None)  # could be index or dict
        time = snapshot.get("time", 0.0)
        duration = snapshot.get("duration", 1.0)

        # find self index: assume first player in players for our agent as in main.py
        self_player = players[0] if players else {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "team": self.team}

        sx, sy = float(self_player.get("x", 0.0)), float(self_player.get("y", 0.0))
        svx, svy = float(self_player.get("vx", 0.0)), float(self_player.get("vy", 0.0))

        bx, by = float(ball.get("x", field_w / 2)), float(ball.get("y", field_h / 2))
        bvx, bvy = float(ball.get("vx", 0.0)), float(ball.get("vy", 0.0))

        # normalize helpers
        def norm_pos(x, y):
            return x / field_w, y / field_h

        def norm_vel(vx, vy):
            return vx / max_speed, vy / max_speed

        def orient(x, y):
            """Mirror coords for team B so state stays consistent left-to-right."""
            if self.team == "A":
                return x, y
            return field_w - x, field_h - y

        def lane_one_hot(y_norm):
            # vertical lanes across pitch width (top-bottom), mirrored by orient()
            bounds = [0.0, 0.23, 0.4, 0.6, 0.77, 1.0]
            zone = 4
            for i in range(len(bounds) - 1):
                if bounds[i] <= y_norm < bounds[i + 1]:
                    zone = i
                    break
            return [1.0 if zone == i else 0.0 for i in range(5)]

        def third_one_hot(x_norm):
            # defensive/middle/attacking thirds along length, mirrored by orient()
            bounds = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
            zone = 2
            for i in range(len(bounds) - 1):
                if bounds[i] <= x_norm < bounds[i + 1]:
                    zone = i
                    break
            return [1.0 if zone == i else 0.0 for i in range(3)]

        # controller flags
        has_ball_self = 0.0
        has_ball_team_other = 0.0
        has_ball_opponent = 0.0

        # controller parsing
        controller_team = None
        controller_idx = None
        if isinstance(controller, dict):
            controller_team = controller.get("team", None)
            controller_idx = controller.get("index", None)
        elif isinstance(controller, int):
            controller_idx = controller
            if 0 <= controller_idx < len(players):
                controller_team = players[controller_idx].get("team", None)
        elif isinstance(controller, str):
            controller_team = controller

        if controller_idx == 0:
            has_ball_self = 1.0
        elif controller_team is not None:
            if controller_team == self.team:
                has_ball_team_other = 1.0
            else:
                has_ball_opponent = 1.0

        # goal positions (assume goals centered on left (x=0) and right (x=field_w))
        if self.team == "A":
            # attacking right goal
            opp_goal_x, opp_goal_y = field_w, field_h / 2
            team_side = 1.0
        else:
            opp_goal_x, opp_goal_y = 0.0, field_h / 2
            team_side = 0.0

        # vectors and distances
        def safe_norm(dx, dy):
            d = math.sqrt(dx * dx + dy * dy) + 1e-6
            return dx / d, dy / d, d

        sb_dx, sb_dy, sb_dist = safe_norm(bx - sx, by - sy)
        bg_dx, bg_dy, bg_dist = safe_norm(opp_goal_x - bx, opp_goal_y - by)

        # nearest teammate/opponent (excluding self)
        def nearest(filter_team: str) -> Tuple[float, float, float, float]:
            best_d = 1e9
            best = (0.0, 0.0, 0.0, 0.0)
            for i, p in enumerate(players):
                if i == 0:
                    continue
                if p.get("team", "").upper() != filter_team.upper():
                    continue
                dx = float(p.get("x", 0.0)) - sx
                dy = float(p.get("y", 0.0)) - sy
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d
                    best = (float(p.get("x", 0.0)),
                            float(p.get("y", 0.0)),
                            float(p.get("vx", 0.0)),
                            float(p.get("vy", 0.0)))
            nx, ny = norm_pos(best[0], best[1])
            nvx, nvy = norm_vel(best[2], best[3])
            return nx, ny, nvx, nvy

        nt_x, nt_y, nt_vx, nt_vy = nearest(self.team)
        opp_team = "B" if self.team == "A" else "A"
        no_x, no_y, no_vx, no_vy = nearest(opp_team)

        # normalize
        nsx, nsy = norm_pos(sx, sy)
        nsvx, nsvy = norm_vel(svx, svy)
        nbx, nby = norm_pos(bx, by)
        nbvx, nbvy = norm_vel(bvx, bvy)
        n_goal_x, n_goal_y = norm_pos(opp_goal_x, opp_goal_y)
        n_sb_dist = math.sqrt(sb_dist) / math.sqrt(field_w * field_w + field_h * field_h)
        n_bg_dist = math.sqrt(bg_dist) / math.sqrt(field_w * field_w + field_h * field_h)

        # zones mirrored to keep B perspective consistent
        osx, osy = orient(sx, sy)
        obx, oby = orient(bx, by)
        vertical_zone = lane_one_hot(osy / field_h if field_h > 0 else 0.0)
        horizontal_third = third_one_hot(osx / field_w if field_w > 0 else 0.0)
        ball_vertical_zone = lane_one_hot(oby / field_h if field_h > 0 else 0.0)
        ball_horizontal_third = third_one_hot(obx / field_w if field_w > 0 else 0.0)

        time_norm = float(time) / float(duration) if duration > 0 else 0.0

        feat = np.array([
            nsx, nsy,
            nsvx, nsvy,
            nbx, nby,
            nbvx, nbvy,
            *vertical_zone,               # 8-12
            *horizontal_third,            # 13-15
            *ball_vertical_zone,          # 16-20
            *ball_horizontal_third,       # 21-23
            has_ball_self,                # 24
            has_ball_team_other,          # 25
            has_ball_opponent,            # 26
            n_goal_x, n_goal_y,           # 27-28
            sb_dx, sb_dy,                 # 29-30
            n_sb_dist,                    # 31
            bg_dx, bg_dy,                 # 32-33
            n_bg_dist,                    # 34
            nt_x, nt_y, nt_vx, nt_vy,     # 35-38
            no_x, no_y, no_vx, no_vy,     # 39-42
            team_side,                    # 43
            1.0 if has_ball_self > 0.5 else 0.0,  # 44
            1.0,                                   # 45 stamina proxy
            time_norm,                              # 46
            1.0                                    # 47 bias
        ], dtype=np.float32)

        # store last_state for learn()
        self.last_state = feat
        return feat

    # ------------- Action Space -------------

    def _build_action_space(self) -> List[Dict[str, Any]]:
        """
        Builds 13 actions:
        0: idle
        1-8: move in 8 directions at normal speed
        9-12: sprint move (N, S, E, W) at sprint speed
        13-15: shoot to goal (low/med/high) — NOTE: we fix to 3 levels by replacing 9-12 scheme to keep n_actions=13
        """
        actions: List[Dict[str, Any]] = []

        # idle
        actions.append(self._make_action(0.0, 0.0, 0.0, None))

        # 8-direction moves (unit circle)
        dirs = [
            (1, 0),   # E
            (-1, 0),  # W
            (0, 1),   # S (assuming y grows downward in screen coords)
            (0, -1),  # N
            (1, 1),   # SE
            (1, -1),  # NE
            (-1, 1),  # SW
            (-1, -1), # NW
        ]
        for dx, dy in dirs:
            ndx, ndy = self._unit(dx, dy)
            actions.append(self._make_action(ndx * self.max_move_speed, ndy * self.max_move_speed, 0.0, None))

        # 4-direction sprint (N, S, E, W) to keep action count reasonable
        sprint_dirs = [
            (1, 0),   # E
            (-1, 0),  # W
            (0, 1),   # S
            (0, -1),  # N
        ]
        for dx, dy in sprint_dirs:
            ndx, ndy = self._unit(dx, dy)
            actions.append(self._make_action(ndx * self.max_move_speed * self.sprint_multiplier,
                                             ndy * self.max_move_speed * self.sprint_multiplier,
                                             0.0, None))

        # Replace last sprint to 3 shooting actions to cap at 13 total.
        # Total currently: 1 + 8 + 4 = 13, we will modify indices 9..12 to be shots instead:
        # To maintain exact 13 actions: we will rebuild with 1 + 8 + 4_sprint OR 3 shots.
        # Better: Keep 13 as 1 idle + 8 moves + 4 actions: 2 sprint + 1 pass + 1 shoot.
        # For clarity and completeness, we reconstruct list deterministically:

        actions = []
        actions.append(self._make_action(0.0, 0.0, 0.0, None))  # 0 idle

        # 8 moves (1..8)
        for dx, dy in dirs:
            ndx, ndy = self._unit(dx, dy)
            actions.append(self._make_action(ndx * self.max_move_speed, ndy * self.max_move_speed, 0.0, None))

        # 9 sprint east
        ndx, ndy = self._unit(1, 0)
        actions.append(self._make_action(ndx * self.max_move_speed * self.sprint_multiplier,
                                         ndy * self.max_move_speed * self.sprint_multiplier,
                                         0.0, None))
        # 10 sprint north
        ndx, ndy = self._unit(0, -1)
        actions.append(self._make_action(ndx * self.max_move_speed * self.sprint_multiplier,
                                         ndy * self.max_move_speed * self.sprint_multiplier,
                                         0.0, None))

        # 11 shoot to opponent goal (medium power)
        actions.append(self._shoot_action(power=0.6))

        # 12 shoot to opponent goal (high power)
        actions.append(self._shoot_action(power=1.0))

        # Now len(actions) == 13
        return actions

    def _unit(self, dx: float, dy: float) -> Tuple[float, float]:
        d = math.sqrt(dx * dx + dy * dy)
        if d < 1e-6:
            return 0.0, 0.0
        return dx / d, dy / d

    def _make_action(self, vx: float, vy: float, kick_power: float, kick_dir: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        return {
            "move_vx": float(vx),
            "move_vy": float(vy),
            "kick_power": float(kick_power),
            "kick_dir": None if kick_dir is None else (float(kick_dir[0]), float(kick_dir[1])),
        }

    def _shoot_action(self, power: float) -> Dict[str, Any]:
        # shoot towards opponent goal based on team side; direction will be re-evaluated in apply_action phase by simulator,
        # but we provide a default dir: +x for A, -x for B.
        default_dir = (1.0, 0.0) if self.team == "A" else (-1.0, 0.0)
        return self._make_action(0.0, 0.0, power, default_dir)

    # ------------- Utils -------------

    def _decay_epsilon(self):
        if self.epsilon_decay_steps <= 0:
            self.epsilon = self.epsilon_end
            return
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.train_steps / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_end)
        )
