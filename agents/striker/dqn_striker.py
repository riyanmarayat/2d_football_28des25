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
    DQN agent dengan observasi egosentris (B di-mirror ke kiri) dan action set 20 aksi balanced.
    """

    ACTION_COUNT = 20

    def __init__(
        self,
        team: str = "A",
        seed: Optional[int] = 42,
        player_index: int = 0,
        state_dim: int = 61,
        n_actions: int = ACTION_COUNT,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        min_buffer_to_learn: int = 1000,
        target_update_interval: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5_000,
        max_move_speed: float = 6.0,
        sprint_multiplier: float = 1.2,
    ):
        self.team = team.upper()
        self.player_index = player_index
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

        self.last_action_idx: Optional[int] = None
        self._last_action_vel: Tuple[float, float] = (0.0, 0.0)
        self._last_action_kick: Tuple[float, Optional[Tuple[float, float]]] = (0.0, None)
        self._last_action_one_hot = np.zeros(self.n_actions, dtype=np.float32)

        self._ego = {
            "sx": 0.0, "sy": 0.0,
            "bx": 0.0, "by": 0.0,
            "ball_carrier_idx": None,
            "opp_goal": (100.0, 37.5),
            "field_w": 100.0, "field_h": 75.0,
            "players": [],
            "best_tm_world": None,
            "opp_team": "B" if self.team == "A" else "A",
        }

    def select_action(self, snapshot: Dict[str, Any]) -> int:
        state_vec = self.extract_features(snapshot)
        self._actions = self._build_action_space()
        if self.rng.random() < self.epsilon:
            action = int(self.rng.integers(0, self.n_actions))
        else:
            with torch.no_grad():
                q = self.policy_net(torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0))
                action = int(torch.argmax(q, dim=1).item())
        self.last_action_idx = action
        self._last_action_one_hot = np.zeros(self.n_actions, dtype=np.float32)
        self._last_action_one_hot[action] = 1.0
        a = self._actions[action]
        mvx, mvy = float(a.get("move_vx", 0.0)), float(a.get("move_vy", 0.0))
        kpow, kdir = float(a.get("kick_power", 0.0)), a.get("kick_dir", None)
        # simpan dalam frame dunia; untuk tim B dibalik arah X
        if self.team == "B":
            mvx = -mvx
            if kdir is not None:
                kdir = (-kdir[0], kdir[1])
        self._last_action_vel = (mvx, mvy)
        self._last_action_kick = (kpow, kdir)
        self._decay_epsilon()
        return action

    def desired_velocity(self, player, ball, field) -> Tuple[float, float]:
        # role guard rails sederhana: GK stay di gawang, CB jaga garis, support off-ball
        vx, vy = self._last_action_vel
        role = str(player.get("role", "")).lower()
        # goal center (world)
        gx = 0.0 if player.get("team", "A").upper() == "B" else field.width
        gy = field.height / 2
        if role == "goalkeeper":
            dx, dy = gx - player["x"], gy - player["y"]
            d = math.hypot(dx, dy) + 1e-6
            return (dx / d) * 4.0, (dy / d) * 4.0
        if "center back" in role:
            # jaga zona defensif
            target_x = field.width * (0.25 if player.get("team", "A").upper() == "A" else 0.75)
            dx, dy = target_x - player["x"], (ball.y - player["y"])
            d = math.hypot(dx, dy) + 1e-6
            return (dx / d) * 5.0, (dy / d) * 5.0
        # jika bukan ball controller dan teammate pegang bola, cari ruang (half-space)
        bc = getattr(ball, "controller", None) if hasattr(ball, "controller") else None
        if bc is not None and bc != self.player_index and player.get("team", "").upper() == ("A" if self.team == "A" else "B"):
            offset = 8.0 if (self.player_index % 2 == 0) else -8.0
            target_y = max(0.0, min(field.height, ball.y + offset))
            target_x = min(field.width * 0.65, ball.x + 6.0) if self.team == "A" else max(field.width * 0.35, ball.x - 6.0)
            dx, dy = target_x - player["x"], target_y - player["y"]
            d = math.hypot(dx, dy) + 1e-6
            return (dx / d) * 5.5, (dy / d) * 5.5
        return vx, vy

    def action_index_to_dict(self, action_idx: int) -> Dict[str, Any]:
        base = dict(self._actions[action_idx])
        if self.team == "B":
            base["move_vx"] = -base.get("move_vx", 0.0)
            if base.get("kick_dir") is not None:
                kx, ky = base["kick_dir"]
                base["kick_dir"] = (-kx, ky)
        return base

    def learn(self, reward: float, next_snapshot: Dict[str, Any], done: bool):
        if self.last_state is None or self.last_action_idx is None:
            return
        next_state_vec = self.extract_features(next_snapshot)
        self.buffer.push(self.last_state, self.last_action_idx, reward, next_state_vec, done)
        if len(self.buffer) < self.min_buffer_to_learn:
            self.last_state = next_state_vec
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states_t)
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

        self.last_state = next_state_vec

    def extract_features(self, snapshot: Dict[str, Any]) -> np.ndarray:
        field_info = snapshot.get("field", {})
        field_w = float(field_info.get("width", 100.0))
        field_h = float(field_info.get("height", 75.0))
        diag = math.sqrt(field_w * field_w + field_h * field_h)
        max_ball_speed = 30.0

        players: List[Dict[str, Any]] = snapshot.get("players", [])
        ball: Dict[str, Any] = snapshot.get("ball", {})
        controller = snapshot.get("ball_controller", None)
        step = float(snapshot.get("step", 0))
        max_steps = float(snapshot.get("duration", 1000) or 1000)

        idx = min(self.player_index, len(players) - 1) if players else 0
        self_player = players[idx] if players else {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "team": self.team}

        def mirror_pos(x, y):
            if self.team == "A":
                return x, y
            return field_w - x, y

        def mirror_vel(vx, vy):
            if self.team == "A":
                return vx, vy
            return -vx, vy

        sx_raw, sy_raw = float(self_player.get("x", 0.0)), float(self_player.get("y", 0.0))
        svx_raw, svy_raw = float(self_player.get("vx", 0.0)), float(self_player.get("vy", 0.0))
        sx, sy = mirror_pos(sx_raw, sy_raw)
        svx, svy = mirror_vel(svx_raw, svy_raw)

        bx_raw, by_raw = float(ball.get("x", field_w / 2)), float(ball.get("y", field_h / 2))
        bvx_raw, bvy_raw = float(ball.get("vx", 0.0)), float(ball.get("vy", 0.0))
        bx, by = mirror_pos(bx_raw, by_raw)
        bvx, bvy = mirror_vel(bvx_raw, bvy_raw)

        def norm(val, denom):
            return val / (denom + 1e-6)

        def rel(dx, dy):
            d = math.sqrt(dx * dx + dy * dy) + 1e-6
            return dx / d, dy / d, d

        ball_dx, ball_dy = bx - sx, by - sy
        ball_dir_x, ball_dir_y, ball_dist = rel(ball_dx, ball_dy)
        goal_dx, goal_dy = field_w - sx, (field_h / 2) - sy
        goal_dir_x, goal_dir_y, goal_dist = rel(goal_dx, goal_dy)

        angle_ball_sin, angle_ball_cos = ball_dir_y, ball_dir_x
        angle_goal_sin, angle_goal_cos = goal_dir_y, goal_dir_x

        ctrl_team = None
        ctrl_idx = None
        if isinstance(controller, dict):
            ctrl_team = controller.get("team", None)
            ctrl_idx = controller.get("index", None)
        elif isinstance(controller, int):
            ctrl_idx = controller
            if 0 <= ctrl_idx < len(players):
                ctrl_team = players[ctrl_idx].get("team", None)
        elif isinstance(controller, str):
            ctrl_team = controller

        bc_none = 1.0 if controller in (None, -1) else 0.0
        bc_me = 1.0 if ctrl_idx == idx else 0.0
        bc_tm = 1.0 if (ctrl_team == self.team and ctrl_idx != idx) else 0.0
        bc_op = 1.0 if (ctrl_team is not None and ctrl_team != self.team) else 0.0

        opp_team = "B" if self.team == "A" else "A"
        nearest_opp = (0.0, 0.0, 1e9)
        opp_count_r8 = 0
        opp_count_r16 = 0
        for p in players:
            if p.get("team", "").upper() != opp_team:
                continue
            px, py = mirror_pos(float(p.get("x", 0.0)), float(p.get("y", 0.0)))
            dx, dy = px - sx, py - sy
            d = math.sqrt(dx * dx + dy * dy)
            if d < nearest_opp[2]:
                nearest_opp = (dx, dy, d)
            if d < 8.0:
                opp_count_r8 += 1
            if d < 16.0:
                opp_count_r16 += 1

        nearest_opp_dx = norm(nearest_opp[0], field_w)
        nearest_opp_dy = norm(nearest_opp[1], field_h)
        nearest_opp_dist = norm(nearest_opp[2], diag)
        max_opp = max(1, sum(1 for p in players if p.get("team", "").upper() == opp_team))
        opp_count_r8_norm = opp_count_r8 / max_opp
        opp_count_r16_norm = opp_count_r16 / max_opp

        best_tm = None
        best_score = -1e9
        for i, p in enumerate(players):
            if i == idx or p.get("team", "").upper() != self.team:
                continue
            px, py = mirror_pos(float(p.get("x", 0.0)), float(p.get("y", 0.0)))
            dx, dy = px - sx, py - sy
            dist = math.sqrt(dx * dx + dy * dy)
            min_opp_d = 1e9
            for q in players:
                if q.get("team", "").upper() != opp_team:
                    continue
                qx, qy = mirror_pos(float(q.get("x", 0.0)), float(q.get("y", 0.0)))
                d_opp = math.sqrt((qx - px) ** 2 + (qy - py) ** 2)
                min_opp_d = min(min_opp_d, d_opp)
            score = -dist + 0.5 * min_opp_d
            if score > best_score:
                best_score = score
                best_tm = (dx, dy, dist, min_opp_d, px, py)

        if best_tm is None:
            best_tm_dx = best_tm_dy = best_tm_dist = 0.0
            best_tm_is_open = 0.0
            best_tm_world = None
        else:
            best_tm_dx = norm(best_tm[0], field_w)
            best_tm_dy = norm(best_tm[1], field_h)
            best_tm_dist = norm(best_tm[2], diag)
            best_tm_is_open = 1.0 if best_tm[3] > 5.0 else 0.0
            bt_xw = best_tm[4] if self.team == "A" else field_w - best_tm[4]
            bt_yw = best_tm[5]
            best_tm_world = (bt_xw, bt_yw)

        tm_ahead = 0
        tm_total = 0
        for i, p in enumerate(players):
            if p.get("team", "").upper() != self.team or i == idx:
                continue
            tm_total += 1
            px, _ = mirror_pos(float(p.get("x", 0.0)), float(p.get("y", 0.0)))
            if px > sx:
                tm_ahead += 1
        teammate_count_ahead_norm = tm_ahead / max(1, tm_total)

        def line_clear(target_x, target_y, radius=2.0):
            min_d = 1e9
            vx, vy = target_x - sx, target_y - sy
            denom = vx * vx + vy * vy + 1e-6
            for p in players:
                if p.get("team", "").upper() != opp_team:
                    continue
                px, py = mirror_pos(float(p.get("x", 0.0)), float(p.get("y", 0.0)))
                t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / denom))
                proj_x = sx + t * vx
                proj_y = sy + t * vy
                d = math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
                min_d = min(min_d, d)
            return min_d > radius

        pass_lane_open = 1.0 if best_tm and line_clear(sx + best_tm[0], sy + best_tm[1], radius=1.5) else 0.0
        shooting_window_open = 1.0 if line_clear(field_w, field_h / 2, radius=2.5) else 0.0
        dribble_lane_open = 1.0 if line_clear(bx, by, radius=1.5) else 0.0

        def lane_y_one_hot(y_norm):
            bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            idx_lane = 4
            for i in range(len(bounds) - 1):
                if bounds[i] <= y_norm < bounds[i + 1]:
                    idx_lane = i
                    break
            return [1.0 if idx_lane == i else 0.0 for i in range(5)]

        def thirds_x_one_hot(x_norm):
            bounds = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
            idx_third = 2
            for i in range(len(bounds) - 1):
                if bounds[i] <= x_norm < bounds[i + 1]:
                    idx_third = i
                    break
            return [1.0 if idx_third == i else 0.0 for i in range(3)]

        lane_y = lane_y_one_hot(norm(sy, field_h))
        thirds_x = thirds_x_one_hot(norm(sx, field_w))

        out_of_bounds = 1.0 if snapshot.get("out_of_bounds", False) else 0.0
        time_frac = min(1.0, step / max_steps)

        feat = np.array([
            norm(sx, field_w), norm(sy, field_h),
            norm(ball_dx, field_w), norm(ball_dy, field_h),
            norm(bvx, max_ball_speed), norm(bvy, max_ball_speed),
            norm(ball_dist, diag), angle_ball_sin, angle_ball_cos,
            norm(goal_dx, field_w), norm(goal_dy, field_h),
            norm(goal_dist, diag), angle_goal_sin, angle_goal_cos,
            bc_none, bc_me, bc_tm, bc_op,
            nearest_opp_dx, nearest_opp_dy, nearest_opp_dist,
            opp_count_r8_norm, opp_count_r16_norm,
            best_tm_dx, best_tm_dy, best_tm_dist, best_tm_is_open,
            teammate_count_ahead_norm,
            pass_lane_open, shooting_window_open, dribble_lane_open,
            *lane_y,
            *thirds_x,
            out_of_bounds,
            *self._last_action_one_hot.tolist(),
            time_frac
        ], dtype=np.float32)

        self.last_state = feat
        self._ego = {
            "sx": sx, "sy": sy,
            "bx": bx, "by": by,
            "ball_carrier_idx": ctrl_idx,
            "opp_goal": (field_w, field_h / 2),
            "field_w": field_w, "field_h": field_h,
            "players": players,
            "best_tm_world": best_tm_world,
            "opp_team": opp_team,
        }
        return feat

    def _build_action_space(self) -> List[Dict[str, Any]]:
        e = self._ego
        sx, sy = e["sx"], e["sy"]
        bx, by = e["bx"], e["by"]
        gx, gy = e["opp_goal"]
        fw = e["field_w"]
        fh = e["field_h"]
        best_tm_world = e.get("best_tm_world")
        players = e.get("players", [])
        opp_team = e.get("opp_team")
        actions: List[Dict[str, Any]] = []

        def mv(dx, dy, speed):
            nd = math.sqrt(dx * dx + dy * dy)
            if nd < 1e-6:
                return 0.0, 0.0
            return dx / nd * speed, dy / nd * speed

        def mirror_pos(x, y):
            if self.team == "A":
                return x, y
            return fw - x, y

        # 0 idle
        actions.append(self._make_action(0.0, 0.0, 0.0, None))

        # 1-8 moves (N,NE,E,SE,S,SW,W,NW) dalam frame ego
        dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        for dx, dy in dirs:
            vx, vy = mv(dx, dy, self.max_move_speed)
            actions.append(self._make_action(vx, vy, 0.0, None))

        # 9 sprint forward (ke gawang lawan)
        vx, vy = mv(gx - sx, gy - sy, self.max_move_speed * self.sprint_multiplier)
        actions.append(self._make_action(vx, vy, 0.0, None))

        # 10 control ball (mendekati bola)
        vx, vy = mv(bx - sx, by - sy, self.max_move_speed * 0.6)
        actions.append(self._make_action(vx, vy, 0.0, None))

        # 11 dribble toward goal + lateral noise
        lat = self.rng.uniform(-0.3, 0.3)
        vx, vy = mv((gx - sx) + lat, (gy - sy) + lat, self.max_move_speed * 0.8)
        actions.append(self._make_action(vx, vy, 0.0, None))

        # 12 pass short
        if best_tm_world is not None:
            txw, tyw = best_tm_world
            tx, ty = (txw if self.team == "A" else fw - txw), tyw
            dx, dy = tx - sx, ty - sy
            mag = math.sqrt(dx * dx + dy * dy) + 1e-6
            actions.append(self._make_action(0.0, 0.0, 0.4, (dx / mag, dy / mag)))
        else:
            actions.append(self._make_action(0.0, 0.0, 0.0, None))

        # 13 through pass (lead forward)
        if best_tm_world is not None:
            txw, tyw = best_tm_world
            lead_xw = txw + 4.0
            tx, ty = (lead_xw if self.team == "A" else fw - lead_xw), tyw
            dx, dy = tx - sx, ty - sy
            mag = math.sqrt(dx * dx + dy * dy) + 1e-6
            actions.append(self._make_action(0.0, 0.0, 0.65, (dx / mag, dy / mag)))
        else:
            actions.append(self._make_action(0.0, 0.0, 0.0, None))

        # 14 lob pass
        if best_tm_world is not None:
            txw, tyw = best_tm_world
            tx, ty = (txw if self.team == "A" else fw - txw), tyw
            dx, dy = tx - sx, ty - sy
            mag = math.sqrt(dx * dx + dy * dy) + 1e-6
            actions.append(self._make_action(0.0, 0.0, 0.9, (dx / mag, dy / mag)))
        else:
            actions.append(self._make_action(0.0, 0.0, 0.0, None))

        # 15 shoot power
        dx, dy = gx - sx, gy - sy
        mag = math.sqrt(dx * dx + dy * dy) + 1e-6
        actions.append(self._make_action(0.0, 0.0, 1.0, (dx / mag, dy / mag)))

        # 16 shoot placed
        actions.append(self._make_action(0.0, 0.0, 0.7, (dx / mag, dy / mag)))

        # 17 tackle (dash ke bola)
        vx, vy = mv(bx - sx, by - sy, self.max_move_speed * self.sprint_multiplier)
        actions.append(self._make_action(vx, vy, 0.0, None))

        # 18 block lane (midpoint bola->gawang)
        midx, midy = (bx + gx) / 2, (by + gy) / 2
        vx, vy = mv(midx - sx, midy - sy, self.max_move_speed * 0.9)
        actions.append(self._make_action(vx, vy, 0.0, None))

        # 19 press (ke ball carrier kalau ada)
        bc_idx = e.get("ball_carrier_idx")
        if bc_idx is not None and 0 <= bc_idx < len(players):
            pc = players[bc_idx]
            px, py = mirror_pos(float(pc.get("x", 0.0)), float(pc.get("y", 0.0)))
        else:
            px, py = bx, by
        vx, vy = mv(px - sx, py - sy, self.max_move_speed * 1.05)
        actions.append(self._make_action(vx, vy, 0.0, None))

        return actions

    def _make_action(self, vx: float, vy: float, kick_power: float, kick_dir: Optional[Tuple[float, float]]) -> Dict[str, Any]:
        return {
            "move_vx": float(vx),
            "move_vy": float(vy),
            "kick_power": float(kick_power),
            "kick_dir": None if kick_dir is None else (float(kick_dir[0]), float(kick_dir[1])),
        }

    def _decay_epsilon(self):
        if self.epsilon_decay_steps <= 0:
            self.epsilon = self.epsilon_end
            return
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.train_steps / self.epsilon_decay_steps) * (self.epsilon_start - self.epsilon_end)
        )
