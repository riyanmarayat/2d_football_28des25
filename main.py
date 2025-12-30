import time
import os
import math
import torch
from typing import List, Dict, Any, Optional
from core.ball import Ball
from core.field import Field
from core.recorder import Recorder
from core.simulator import Simulator, compute_agent_reward
from render.pygame_renderer import PygameRenderer
from render.video_exporter import VideoExporter
from agents.striker.dqn_striker import DQNStriker
from agents.dqn_roles import (
    DQNGoalkeeperAgent, DQNCenterBackAgent, DQNRightFullbackAgent, DQNLeftFullbackAgent,
    DQNCentralMidfielderAgent, DQNRightMidfielderAgent, DQNLeftMidfielderAgent,
    DQNRightWingerAgent, DQNLeftWingerAgent,
)

FPS = 15
DURATION_STEPS = 1000
try:
    NUM_EPISODES = int(input("Berapa episode training? [default 1]: ") or "1")
except Exception:
    NUM_EPISODES = 1
CHECKPOINT_DIR = "checkpoints"

def save_team_checkpoint(team_name: str, agents: List, side: str):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    payload = []
    for ag in agents:
        if getattr(ag, "team", "").upper() == team_name.upper() and getattr(ag, "side", "left") == side:
            payload.append({
                "state_dict": ag.policy_net.state_dict(),
            })
    path = os.path.join(CHECKPOINT_DIR, f"{team_name.upper()}_{side}.pth")
    torch.save(payload, path)
    print(f"Checkpoint saved for team {team_name} side {side} -> {path}")

def load_team_checkpoint(team_name: str, agents: List, side: str):
    path = os.path.join(CHECKPOINT_DIR, f"{team_name.upper()}_{side}.pth")
    if not os.path.exists(path):
        return
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
        return
    # apply sequentially to agents of that team
    idx = 0
    for ag in agents:
        if getattr(ag, "team", "").upper() != team_name.upper() or getattr(ag, "side", "left") != side:
            continue
        if idx < len(payload):
            ag.policy_net.load_state_dict(payload[idx]["state_dict"])
            ag.target_net.load_state_dict(payload[idx]["state_dict"])
            idx += 1
    print(f"Loaded checkpoint for team {team_name} side {side} from {path}")

# Fallback jika apply_action tidak menggerakkan pemain
def apply_action_compat(simulator, player, action, dt):
    vx = float(action.get("move_vx", 0.0))
    vy = float(action.get("move_vy", 0.0))
    player["vx"] = vx
    player["vy"] = vy
    player["x"] += vx * dt
    player["y"] += vy * dt
    # clamp dalam lapangan
    fw = simulator.field.width
    fh = simulator.field.height
    player["x"] = max(0.0, min(fw, player["x"]))
    player["y"] = max(0.0, min(fh, player["y"]))
    # proses tendangan jika power > 0 dan pemain pengontrol bola
    kp = float(action.get("kick_power", 0.0))
    kdir = action.get("kick_dir", None)
    if kp > 0.0 and kdir is not None:
        controller = simulator.ball_controller
        is_self = False
        if isinstance(controller, int):
            is_self = (controller == 0)
        elif isinstance(controller, dict):
            is_self = (controller.get("index", None) == 0)
        else:
            dx = simulator.ball.x - player["x"]
            dy = simulator.ball.y - player["y"]
            is_self = (dx*dx + dy*dy) <= 1.0
        if is_self:
            dx, dy = float(kdir[0]), float(kdir[1])
            mag = (dx*dx + dy*dy) ** 0.5
            if mag > 1e-6:
                dx /= mag; dy /= mag
            ball_speed = 25.0 * max(0.0, min(1.0, kp))
            simulator.ball.vx = dx * ball_speed
            simulator.ball.vy = dy * ball_speed

class StatsTracker:
    def __init__(self, field: Field, fps: int, team_left: str, team_right: str):
        self.field = field
        self.dt = 1.0 / float(fps)
        self.side_to_team = {"left": team_left, "right": team_right}
        self.window_player_stats: Dict[int, Dict[str, float]] = {}
        self.window_team_stats: Dict[str, Dict[str, float]] = {}
        self.window_episode_count = 0
        self.player_meta: Dict[int, Dict[str, Any]] = {}
        self.goal_y_top = (self.field.height - 7.32) / 2
        self.goal_y_bottom = (self.field.height + 7.32) / 2

    def start_episode(self, players: List[Dict[str, Any]]):
        self.player_meta = {
            i: {
                "team": p.get("team", "?"),
                "role": p.get("role", "?"),
                "side": p.get("side", "left")
            }
            for i, p in enumerate(players)
        }
        self.player_stats = {i: {"dribbles": 0, "tackles": 0, "shots_on": 0, "shots_off": 0, "goals": 0}
                             for i in self.player_meta}
        teams = {meta["team"] for meta in self.player_meta.values()}
        self.team_stats = {tm: {"possession_time": 0.0, "goals_for": 0, "goals_against": 0} for tm in teams}
        self.control_anchor: Dict[int, tuple] = {}
        self.last_controller: Optional[int] = None
        self.total_time = 0.0
        self._goal_recorded = False

    def record_step(self, old_state: Dict[str, Any], new_state: Dict[str, Any], actions: List[Dict[str, Any]], dt: float):
        players_new = new_state.get("players", [])
        players_old = old_state.get("players", players_new) if old_state else players_new
        prev_ctrl = old_state.get("ball_controller") if old_state else None
        curr_ctrl = new_state.get("ball_controller")
        self.total_time += dt
        self._update_possession(curr_ctrl, players_new, dt)
        self._update_control(prev_ctrl, curr_ctrl, players_new)
        self._update_shots(prev_ctrl, actions, players_old)
        self._update_goal(new_state)

    def finish_episode(self, episode_idx: int):
        self._add_to_window()
        episode_summary = self._build_episode_summary(episode_idx)
        rolling_summary: List[str] = []
        if self.window_episode_count >= 100:
            rolling_summary = self._build_window_summary(episode_idx)
            self.window_episode_count = 0
            self.window_player_stats = {}
            self.window_team_stats = {}
        return episode_summary, rolling_summary

    def _update_possession(self, controller, players, dt: float):
        if controller is None or controller >= len(players):
            return
        team = players[controller].get("team")
        if team in self.team_stats:
            self.team_stats[team]["possession_time"] += dt

    def _update_control(self, prev_ctrl, curr_ctrl, players):
        if prev_ctrl is not None and prev_ctrl != curr_ctrl and prev_ctrl in self.control_anchor and prev_ctrl < len(players):
            start_x, start_y = self.control_anchor.get(prev_ctrl, (players[prev_ctrl].get("x", 0.0), players[prev_ctrl].get("y", 0.0)))
            px = players[prev_ctrl].get("x", 0.0)
            py = players[prev_ctrl].get("y", 0.0)
            if math.hypot(px - start_x, py - start_y) >= 1.5:
                self.player_stats[prev_ctrl]["dribbles"] += 1
        if prev_ctrl is not None and curr_ctrl is not None and prev_ctrl != curr_ctrl:
            if prev_ctrl < len(players) and curr_ctrl < len(players):
                prev_team = players[prev_ctrl].get("team")
                curr_team = players[curr_ctrl].get("team")
                if prev_team and curr_team and prev_team != curr_team:
                    dx = players[curr_ctrl].get("x", 0.0) - players[prev_ctrl].get("x", 0.0)
                    dy = players[curr_ctrl].get("y", 0.0) - players[prev_ctrl].get("y", 0.0)
                    if math.hypot(dx, dy) <= 2.0:
                        self.player_stats[curr_ctrl]["tackles"] += 1
        if curr_ctrl is not None and curr_ctrl < len(players) and curr_ctrl != prev_ctrl:
            self.control_anchor[curr_ctrl] = (players[curr_ctrl].get("x", 0.0), players[curr_ctrl].get("y", 0.0))
        self.last_controller = curr_ctrl

    def _classify_shot(self, action: Dict[str, Any], player: Dict[str, Any]) -> Optional[str]:
        power = float(action.get("kick_power", 0.0) or 0.0)
        kdir = action.get("kick_dir")
        if power <= 0.0 or not kdir:
            return None
        dir_x, dir_y = float(kdir[0]), float(kdir[1])
        dir_norm = math.hypot(dir_x, dir_y)
        if dir_norm < 1e-6:
            return None
        side = player.get("side", "left")
        goal_x = self.field.width if side == "left" else 0.0
        goal_vec_x = goal_x - float(player.get("x", 0.0))
        goal_vec_y = (self.field.height / 2) - float(player.get("y", 0.0))
        goal_norm = math.hypot(goal_vec_x, goal_vec_y)
        if goal_norm < 1e-6:
            return None
        cos_theta = (dir_x * goal_vec_x + dir_y * goal_vec_y) / (dir_norm * goal_norm + 1e-6)
        if cos_theta < 0.7:
            return None
        if abs(dir_x) < 1e-5:
            return None
        t = (goal_x - float(player.get("x", 0.0))) / dir_x
        if t <= 0:
            return None
        y_at_goal = float(player.get("y", 0.0)) + dir_y * t
        if self.goal_y_top <= y_at_goal <= self.goal_y_bottom:
            return "on"
        return "off"

    def _update_shots(self, controller_old, actions: List[Dict[str, Any]], players_old: List[Dict[str, Any]]):
        if controller_old is None:
            return
        if controller_old >= len(actions) or controller_old >= len(players_old):
            return
        shot_type = self._classify_shot(actions[controller_old], players_old[controller_old])
        if shot_type is None:
            return
        if shot_type == "on":
            self.player_stats[controller_old]["shots_on"] += 1
        else:
            self.player_stats[controller_old]["shots_off"] += 1

    def _update_goal(self, new_state: Dict[str, Any]):
        goal_side = new_state.get("last_goal")
        if not goal_side or self._goal_recorded:
            return
        scoring_side = "left" if goal_side == "right" else "right"
        scoring_team = self.side_to_team.get(scoring_side)
        conceded_team = self.side_to_team.get("right" if scoring_side == "left" else "left")
        if scoring_team in self.team_stats:
            self.team_stats[scoring_team]["goals_for"] += 1
        if conceded_team in self.team_stats:
            self.team_stats[conceded_team]["goals_against"] += 1
        scorer_idx = new_state.get("last_goal_scorer")
        if scorer_idx is None:
            scorer_idx = new_state.get("last_touch")
        if scorer_idx is not None and scorer_idx in self.player_stats:
            self.player_stats[scorer_idx]["goals"] += 1
            if self.player_stats[scorer_idx]["shots_on"] < self.player_stats[scorer_idx]["goals"]:
                self.player_stats[scorer_idx]["shots_on"] += 1
        self._goal_recorded = True

    def _add_to_window(self):
        self.window_episode_count += 1
        for idx, stats in self.player_stats.items():
            agg = self.window_player_stats.setdefault(idx, {"dribbles": 0, "tackles": 0, "shots_on": 0, "shots_off": 0, "goals": 0})
            for key in ("dribbles", "tackles", "shots_on", "shots_off", "goals"):
                agg[key] += stats.get(key, 0)
        for team, stats in self.team_stats.items():
            agg = self.window_team_stats.setdefault(team, {"possession_time": 0.0, "goals_for": 0, "goals_against": 0, "total_time": 0.0})
            agg["possession_time"] += stats.get("possession_time", 0.0)
            agg["goals_for"] += stats.get("goals_for", 0)
            agg["goals_against"] += stats.get("goals_against", 0)
            agg["total_time"] += self.total_time

    def _build_episode_summary(self, episode_idx: int) -> List[str]:
        total_time = max(self.total_time, 1e-6)
        lines = [f"=== Ringkasan Episode {episode_idx} ==="]
        for side in ("left", "right"):
            team = self.side_to_team.get(side)
            if team and team in self.team_stats:
                ts = self.team_stats[team]
                poss_pct = (ts["possession_time"] / total_time) * 100.0
                lines.append(f"Tim {team} (side {side}): possession {poss_pct:.1f}%, goal {ts['goals_for']}, kebobolan {ts['goals_against']}")
        lines.append("Pemain:")
        for idx in sorted(self.player_stats.keys()):
            meta = self.player_meta.get(idx, {})
            st = self.player_stats[idx]
            label = f"p{idx} [{meta.get('team', '?')} {meta.get('role', '?')}]"
            lines.append(f"- {label}: dribble {st['dribbles']}, tackle {st['tackles']}, on-target {st['shots_on']}, off-target {st['shots_off']}, goals {st['goals']}")
        return lines

    def _player_label(self, idx: int) -> str:
        meta = self.player_meta.get(idx, {})
        return f"p{idx}({meta.get('team', '?')},{meta.get('role', '?')})"

    def _top_players(self, key: str, top_n: int = 3) -> str:
        ranked = sorted(self.window_player_stats.items(), key=lambda kv: kv[1].get(key, 0), reverse=True)
        ranked = [r for r in ranked if r[1].get(key, 0) > 0][:top_n]
        if not ranked:
            return "tidak ada"
        parts = [f"{self._player_label(idx)}={stat.get(key, 0)}" for idx, stat in ranked]
        return ", ".join(parts)

    def _build_window_summary(self, episode_idx: int) -> List[str]:
        start_ep = episode_idx - self.window_episode_count + 1
        lines = [f"=== Ringkasan Episode {start_ep}-{episode_idx} (setiap 100 ep) ==="]
        for side in ("left", "right"):
            team = self.side_to_team.get(side)
            if team and team in self.window_team_stats:
                ts = self.window_team_stats[team]
                total_time = max(ts.get("total_time", 0.0), 1e-6)
                poss_pct = (ts.get("possession_time", 0.0) / total_time) * 100.0
                lines.append(f"Tim {team}: rata2 possession {poss_pct:.1f}%, total goal {ts.get('goals_for', 0)}, kebobolan {ts.get('goals_against', 0)}")
        lines.append(f"Top dribble: {self._top_players('dribbles')}")
        lines.append(f"Top tackle: {self._top_players('tackles')}")
        lines.append(f"Top shot on target: {self._top_players('shots_on')}")
        lines.append(f"Top goal: {self._top_players('goals')}")
        return lines

field = Field(width=100, height=75)

team_formations = {
    "A": [("Goalkeeper", (5, 37.5)),
          ("Left Fullback", (18, 20)), ("Center Back", (18, 32)), ("Center Back", (18, 45)), ("Right Fullback", (18, 57)),
          ("Left Midfielder", (35, 20)), ("Central Midfielder", (35, 32)), ("Central Midfielder", (35, 45)), ("Right Midfielder", (35, 57)),
          ("Striker", (42, 32)), ("Striker", (42, 45))],
    "B": [("Goalkeeper", (5, 37.5)),
          ("Right Fullback", (18, 20)), ("Center Back", (18, 32)), ("Center Back", (18, 45)), ("Left Fullback", (18, 57)),
          ("Central Midfielder", (35, 25)), ("Central Midfielder", (35, 37.5)), ("Central Midfielder", (35, 50)),
          ("Right Winger", (40, 25)), ("Striker", (45, 37.5)), ("Left Winger", (40, 50))],
    "C": [("Goalkeeper", (5, 37.5)),
          ("Left Fullback", (18, 25)), ("Center Back", (18, 37.5)), ("Right Fullback", (18, 50)),
          ("Central Midfielder", (35, 25)), ("Central Midfielder", (35, 50)), ("Central Midfielder", (40, 37.5)),
          ("Right Winger", (40, 25)), ("Left Winger", (40, 50)),
          ("Striker", (45, 37.5))],
    "D": [("Goalkeeper", (5, 37.5)),
          ("Center Back", (15, 25)), ("Center Back", (15, 37.5)), ("Center Back", (15, 50)),
          ("Right Midfielder", (30, 22)), ("Left Midfielder", (30, 53)), ("Central Midfielder", (35, 32)), ("Central Midfielder", (35, 45)), ("Central Midfielder", (40, 37.5)),
          ("Striker", (45, 32)), ("Striker", (45, 45))]
}

team_left = input("Pilih tim kiri (A/B/C/D) [default A]: ").strip().upper() or "A"
team_right = input("Pilih tim kanan (A/B/C/D) [default B]: ").strip().upper() or "B"
stats_tracker = StatsTracker(field, FPS, team_left, team_right)

def build_players(team_name: str, side: str) -> List[Dict[str, Any]]:
    tpl = []
    base_positions = team_formations.get(team_name, team_formations["A"])
    for role, (x, y) in base_positions:
        px = x if side == "left" else field.width - x
        tpl.append({"team": team_name, "role": role, "x": px, "y": y, "vx": 0.0, "vy": 0.0, "side": side,
                    "home_x": px, "home_y": y})
    return tpl

for ep in range(NUM_EPISODES):
    # rebuild players and agents per episode to keep indices/side aligned
    players = build_players(team_left, "left") + build_players(team_right, "right")
    stats_tracker.start_episode(players)
    role_cls = {
        'goalkeeper': DQNGoalkeeperAgent,
        'center back': DQNCenterBackAgent,
        'right fullback': DQNRightFullbackAgent,
        'left fullback': DQNLeftFullbackAgent,
        'central midfielder': DQNCentralMidfielderAgent,
        'right midfielder': DQNRightMidfielderAgent,
        'left midfielder': DQNLeftMidfielderAgent,
        'right winger': DQNRightWingerAgent,
        'left winger': DQNLeftWingerAgent,
        'striker': DQNStriker,
    }
    agents = []
    for idx, p in enumerate(players):
        role = p['role'].lower()
        team = p['team']
        side = p.get("side", "left")
        cls = role_cls.get(role, DQNCentralMidfielderAgent)
        agents.append(cls(team=team, player_index=idx, side=side, seed=idx + ep * 100))

    # Load checkpoints untuk sisi kiri/kanan jika ada
    load_team_checkpoint(team_left, agents, "left")
    load_team_checkpoint(team_right, agents, "right")

    ball = Ball(field_width=field.width, field_height=field.height)
    recorder = Recorder()
    renderer = PygameRenderer(width=field.width, height=field.height, scale=10, show_horizontal_zones=False, show_vertical_zones=False)
    exporter = VideoExporter(f"simulation_ep{ep+1}.mp4", fps=FPS)

    simulator = Simulator(agents, players, ball, field, recorder, fps=FPS)

    old_state = simulator.snapshot()
    total_rewards = [0.0 for _ in agents]
    log_path = f"episode_{ep+1}_log.txt"
    log_lines = []

    # Initialize last_state for each agent
    for ag in agents:
        ag.extract_features(old_state)

    for step in range(DURATION_STEPS):
        dt = 1.0 / FPS

        actions = []
        for i, (agent, player) in enumerate(zip(agents, players)):
            action_idx = agent.select_action(old_state)
            action_dict = agent.action_index_to_dict(action_idx)
            actions.append(action_dict)
            simulator.apply_action(player, action_dict, dt)

        simulator.step(dt=dt)

        new_state = simulator.snapshot()
        done = bool(simulator.last_goal or simulator.out_of_bounds or simulator.offside)

        for i, agent in enumerate(agents):
            reward = compute_agent_reward(simulator, agent, old_state, new_state, actions[i])
            if agent.last_action_idx is not None:
                agent.learn(reward, new_state, done)
            total_rewards[i] += reward

        stats_tracker.record_step(old_state, new_state, actions, dt)

        # log aksi dan state ringkas
        action_log = []
        for i, act in enumerate(actions):
            action_log.append(f"p{i}:{act}")
        state_log = []
        for i, pl in enumerate(players):
            state_log.append(f"p{i}({pl['team']}{pl.get('side','')},{pl['role']}):({pl['x']:.2f},{pl['y']:.2f}) v=({pl.get('vx',0):.2f},{pl.get('vy',0):.2f})")

        line = (f"[Ep {ep+1}] Step {step+1}/{DURATION_STEPS} r0:{total_rewards[0]:.3f} "
                f"actions: {'; '.join(action_log)} | state: {'; '.join(state_log)}")
        log_lines.append(line)
        # terminal ringkas: reward dan pos pemain pertama
        print(f"[Ep {ep+1}] Step {step+1}/{DURATION_STEPS} r0:{total_rewards[0]:.3f} "
              f"p0=({players[0]['x']:.2f},{players[0]['y']:.2f}) p1=({players[1]['x']:.2f},{players[1]['y']:.2f})")
        old_state = new_state

        frame = renderer.render(new_state)
        exporter.add_frame(frame)

        if done:
            break

    exporter.export()
    renderer.quit()
    episode_summary, rolling_summary = stats_tracker.finish_episode(ep+1)
    print(f"Episode {ep+1} saved to {exporter.path}")
    print(f"Episode rewards (sum): {total_rewards}")
    print("\n".join(episode_summary))
    if rolling_summary:
        print("\n".join(rolling_summary))
    # simpan log
    log_lines.append("")
    log_lines.extend(episode_summary)
    if rolling_summary:
        log_lines.append("")
        log_lines.extend(rolling_summary)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Episode log saved to {log_path}")

    # Save checkpoints per team
    save_team_checkpoint(team_left, agents, "left")
    save_team_checkpoint(team_right, agents, "right")
    if (ep + 1) % 100 == 0:
        save_team_checkpoint(f"{team_left}_ep{ep+1}", agents, "left")
        save_team_checkpoint(f"{team_right}_ep{ep+1}", agents, "right")
