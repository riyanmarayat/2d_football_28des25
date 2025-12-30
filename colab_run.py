"""
Headless training runner for Google Colab / servers (non-interactive).
"""

import os
import math
import json
import csv
import argparse
from typing import List, Dict, Any, Optional

import torch

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
from core.scenario_bank import sample_episode_scenario


class StatsTracker:
    def __init__(self, field: Field, fps: int, team_left: str, team_right: str):
        self.field = field
        self.dt = 1.0 / float(fps)
        self.side_to_team = {"left": team_left, "right": team_right}
        self.player_meta: Dict[int, Dict[str, Any]] = {}
        self.window_player_stats_100: Dict[int, Dict[str, float]] = {}
        self.window_team_stats_100: Dict[str, Dict[str, float]] = {}
        self.window_episode_count_100 = 0
        self.window_player_stats_1000: Dict[int, Dict[str, float]] = {}
        self.window_team_stats_1000: Dict[str, Dict[str, float]] = {}
        self.window_episode_count_1000 = 0
        self.goal_y_top = (self.field.height - 7.32) / 2
        self.goal_y_bottom = (self.field.height + 7.32) / 2
        self.pending_passes: List[Dict[str, Any]] = []
        self.pending_shots: List[Dict[str, Any]] = []

    def start_episode(self, players: List[Dict[str, Any]], scenario_name: str = "default", scenario_variant: int = 0):
        self.player_meta = {
            i: {
                "team": p.get("team", "?"),
                "role": p.get("role", "?"),
                "side": p.get("side", "left")
            }
            for i, p in enumerate(players)
        }
        self.scenario_name = scenario_name
        self.scenario_variant = scenario_variant
        self.player_stats = {
            i: {
                "dribbles": 0, "tackles": 0, "blocks": 0,
                "passes": 0, "passes_completed": 0,
                "shots_on": 0, "shots_off": 0,
                "goals": 0, "clears": 0, "saves": 0
            } for i in self.player_meta
        }
        teams = {meta["team"] for meta in self.player_meta.values()}
        self.team_stats = {tm: {"possession_time": 0.0, "goals_for": 0, "goals_against": 0} for tm in teams}
        self.control_anchor: Dict[int, tuple] = {}
        self.last_controller: Optional[int] = None
        self.total_time = 0.0
        self._goal_recorded = False
        self.pending_passes = []
        self.pending_shots = []

    def record_step(self, old_state: Dict[str, Any], new_state: Dict[str, Any], actions: List[Dict[str, Any]], dt: float):
        players_new = new_state.get("players", [])
        players_old = old_state.get("players", players_new) if old_state else players_new
        prev_ctrl = old_state.get("ball_controller") if old_state else None
        curr_ctrl = new_state.get("ball_controller")
        self.total_time += dt

        self._detect_kick(prev_ctrl, actions, players_old)
        self._update_possession(curr_ctrl, players_new, dt)
        self._update_control(prev_ctrl, curr_ctrl, players_new)
        self._update_goal(new_state)
        self._resolve_pending_shots(curr_ctrl, players_new, dt)
        self._resolve_pending_passes(curr_ctrl, players_new, dt)

    def finish_episode(self, episode_idx: int):
        ep_rows = self._build_episode_rows(episode_idx)
        window_rows_100 = []
        window_rows_1000 = []
        ready_100, ready_1000 = self._add_to_windows()
        if ready_100:
            start = episode_idx - 100 + 1
            window_rows_100 = self._build_window_rows("window100", start, episode_idx,
                                                      self.window_player_stats_100, self.window_team_stats_100)
            self.window_player_stats_100 = {}
            self.window_team_stats_100 = {}
            self.window_episode_count_100 = 0
        if ready_1000:
            start = episode_idx - 1000 + 1
            window_rows_1000 = self._build_window_rows("window1000", start, episode_idx,
                                                       self.window_player_stats_1000, self.window_team_stats_1000)
            self.window_player_stats_1000 = {}
            self.window_team_stats_1000 = {}
            self.window_episode_count_1000 = 0
        return ep_rows, window_rows_100, window_rows_1000

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

    def _detect_kick(self, controller_old, actions: List[Dict[str, Any]], players_old: List[Dict[str, Any]]):
        if controller_old is None:
            return
        if controller_old >= len(actions) or controller_old >= len(players_old):
            return
        act = actions[controller_old]
        power = float(act.get("kick_power", 0.0) or 0.0)
        kdir = act.get("kick_dir")
        if power <= 0.0 or not kdir:
            return
        player = players_old[controller_old]
        shot_type = self._classify_shot(act, player)
        if shot_type == "on":
            self.player_stats[controller_old]["shots_on"] += 1
            defending_team = self._opponent_team(player.get("team"))
            self.pending_shots.append({
                "shooter": controller_old,
                "att_team": player.get("team"),
                "def_team": defending_team,
                "time_left": 1.0,
                "on_target": True
            })
            return
        if shot_type == "off":
            self.player_stats[controller_old]["shots_off"] += 1
            return
        dir_x, dir_y = float(kdir[0]), float(kdir[1])
        best_cos, best_dist = self._best_teammate_alignment(controller_old, players_old, dir_x, dir_y)
        if best_cos >= 0.5 and best_dist <= 35.0:
            self.player_stats[controller_old]["passes"] += 1
            self.pending_passes.append({
                "passer": controller_old,
                "team": player.get("team"),
                "time_left": 2.5
            })
        else:
            self.player_stats[controller_old]["clears"] += 1

    def _best_teammate_alignment(self, idx: int, players: List[Dict[str, Any]], dir_x: float, dir_y: float):
        best_cos = -1.0
        best_dist = 1e9
        if abs(dir_x) < 1e-9 and abs(dir_y) < 1e-9:
            return best_cos, best_dist
        norm = math.hypot(dir_x, dir_y) + 1e-6
        dir_x /= norm
        dir_y /= norm
        if idx >= len(players):
            return best_cos, best_dist
        me_team = players[idx].get("team")
        sx = players[idx].get("x", 0.0)
        sy = players[idx].get("y", 0.0)
        for j, p in enumerate(players):
            if j == idx or p.get("team") != me_team:
                continue
            dx = p.get("x", 0.0) - sx
            dy = p.get("y", 0.0) - sy
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            ndx = dx / dist
            ndy = dy / dist
            cos_val = ndx * dir_x + ndy * dir_y
            if cos_val > best_cos:
                best_cos = cos_val
                best_dist = dist
        return best_cos, best_dist

    def _resolve_pending_passes(self, curr_ctrl, players, dt: float):
        next_pending = []
        for item in self.pending_passes:
            item["time_left"] -= dt
            if curr_ctrl is not None and curr_ctrl < len(players):
                if players[curr_ctrl].get("team") == item["team"] and curr_ctrl != item["passer"]:
                    self.player_stats[item["passer"]]["passes_completed"] += 1
                    continue
                if players[curr_ctrl].get("team") != item["team"]:
                    continue
            if item["time_left"] > 0:
                next_pending.append(item)
        self.pending_passes = next_pending

    def _resolve_pending_shots(self, curr_ctrl, players, dt: float):
        next_pending = []
        for item in self.pending_shots:
            item["time_left"] -= dt
            resolved = False
            if self._goal_recorded:
                resolved = True
            elif curr_ctrl is not None and curr_ctrl < len(players):
                ctrl_team = players[curr_ctrl].get("team")
                if item["on_target"] and ctrl_team == item["def_team"]:
                    role = str(players[curr_ctrl].get("role", "")).lower()
                    if "goalkeeper" in role:
                        self.player_stats[curr_ctrl]["saves"] += 1
                    else:
                        self.player_stats[curr_ctrl]["blocks"] += 1
                    resolved = True
            if not resolved and item["time_left"] > 0:
                next_pending.append(item)
        self.pending_shots = next_pending

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
        t = (goal_x - float(player.get("x", 0.0))) / dir_x if abs(dir_x) > 1e-6 else None
        if t is None or t <= 0:
            return None
        y_at_goal = float(player.get("y", 0.0)) + dir_y * t
        if self.goal_y_top <= y_at_goal <= self.goal_y_bottom:
            return "on"
        return "off"

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
        self.pending_shots.clear()
        self._goal_recorded = True

    def _add_to_windows(self):
        self.window_episode_count_100 += 1
        self.window_episode_count_1000 += 1
        for idx, stats in self.player_stats.items():
            for target in (self.window_player_stats_100, self.window_player_stats_1000):
                agg = target.setdefault(idx, {k: 0 for k in stats})
                for key, val in stats.items():
                    agg[key] += val
        for team, stats in self.team_stats.items():
            for target in (self.window_team_stats_100, self.window_team_stats_1000):
                agg = target.setdefault(team, {"possession_time": 0.0, "goals_for": 0, "goals_against": 0, "total_time": 0.0})
                agg["possession_time"] += stats.get("possession_time", 0.0)
                agg["goals_for"] += stats.get("goals_for", 0)
                agg["goals_against"] += stats.get("goals_against", 0)
                agg["total_time"] += self.total_time
        return (self.window_episode_count_100 >= 100, self.window_episode_count_1000 >= 1000)

    def _build_episode_rows(self, episode_idx: int) -> List[Dict[str, Any]]:
        total_time = max(self.total_time, 1e-6)
        rows: List[Dict[str, Any]] = []
        for side in ("left", "right"):
            team = self.side_to_team.get(side)
            if team and team in self.team_stats:
                ts = self.team_stats[team]
                poss_pct = (ts["possession_time"] / total_time) * 100.0
                team_pass = self._aggregate_pass_for_team(team, self.player_stats)
                rows.append({
                    "category": "team",
                    "window": "",
                    "episode": episode_idx,
                    "scenario": self.scenario_name,
                    "scenario_variant": self.scenario_variant,
                    "entity": team,
                    "team": team,
                    "role": "",
                    "possession_pct": poss_pct,
                    "dribbles": "",
                    "tackles": "",
                    "blocks": "",
                    "passes": team_pass["passes"],
                    "passes_completed": team_pass["passes_completed"],
                    "pass_accuracy": team_pass["pass_accuracy"],
                    "shots_on": "",
                    "shots_off": "",
                    "goals": ts["goals_for"],
                    "clears": "",
                    "saves": "",
                    "goals_for": ts["goals_for"],
                    "goals_against": ts["goals_against"]
                })
        for idx in sorted(self.player_stats.keys()):
            meta = self.player_meta.get(idx, {})
            st = self.player_stats[idx]
            acc = (st["passes_completed"] / st["passes"] * 100.0) if st["passes"] > 0 else 0.0
            rows.append({
                "category": "player",
                "window": "",
                "episode": episode_idx,
                "scenario": self.scenario_name,
                "scenario_variant": self.scenario_variant,
                "entity": f"p{idx}",
                "team": meta.get("team", "?"),
                "role": meta.get("role", "?"),
                "possession_pct": "",
                "dribbles": st["dribbles"],
                "tackles": st["tackles"],
                "blocks": st["blocks"],
                "passes": st["passes"],
                "passes_completed": st["passes_completed"],
                "pass_accuracy": acc,
                "shots_on": st["shots_on"],
                "shots_off": st["shots_off"],
                "goals": st["goals"],
                "clears": st["clears"],
                "saves": st["saves"],
                "goals_for": "",
                "goals_against": ""
            })
        return rows

    def _build_window_rows(self, category: str, start_ep: int, end_ep: int,
                           player_stats: Dict[int, Dict[str, float]], team_stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        window_label = f"{start_ep}-{end_ep}"
        for team, ts in team_stats.items():
            total_time = max(ts.get("total_time", 0.0), 1e-6)
            poss_pct = (ts.get("possession_time", 0.0) / total_time) * 100.0
            team_pass = self._aggregate_pass_for_team(team, player_stats)
            rows.append({
                "category": category,
                "window": window_label,
                "episode": end_ep,
                "scenario": "",
                "scenario_variant": "",
                "entity": team,
                "team": team,
                "role": "",
                "possession_pct": poss_pct,
                "dribbles": "",
                "tackles": "",
                "blocks": "",
                "passes": team_pass["passes"],
                "passes_completed": team_pass["passes_completed"],
                "pass_accuracy": team_pass["pass_accuracy"],
                "shots_on": "",
                "shots_off": "",
                "goals": ts.get("goals_for", 0),
                "clears": "",
                "saves": "",
                "goals_for": ts.get("goals_for", 0),
                "goals_against": ts.get("goals_against", 0)
            })
        for idx, st in player_stats.items():
            meta = self.player_meta.get(idx, {})
            acc = (st.get("passes_completed", 0) / st.get("passes", 0) * 100.0) if st.get("passes", 0) > 0 else 0.0
            rows.append({
                "category": category,
                "window": window_label,
                "episode": end_ep,
                "scenario": "",
                "scenario_variant": "",
                "entity": f"p{idx}",
                "team": meta.get("team", "?"),
                "role": meta.get("role", "?"),
                "possession_pct": "",
                "dribbles": st.get("dribbles", 0),
                "tackles": st.get("tackles", 0),
                "blocks": st.get("blocks", 0),
                "passes": st.get("passes", 0),
                "passes_completed": st.get("passes_completed", 0),
                "pass_accuracy": acc,
                "shots_on": st.get("shots_on", 0),
                "shots_off": st.get("shots_off", 0),
                "goals": st.get("goals", 0),
                "clears": st.get("clears", 0),
                "saves": st.get("saves", 0),
                "goals_for": "",
                "goals_against": ""
            })
        return rows

    def _aggregate_pass_for_team(self, team: str, pstats: Dict[int, Dict[str, float]]):
        passes = passes_completed = 0
        for idx, st in pstats.items():
            meta = self.player_meta.get(idx, {})
            if meta.get("team") != team:
                continue
            passes += st.get("passes", 0)
            passes_completed += st.get("passes_completed", 0)
        acc = (passes_completed / passes * 100.0) if passes > 0 else 0.0
        return {"passes": passes, "passes_completed": passes_completed, "pass_accuracy": acc}

    def _opponent_team(self, team: str):
        for side, t in self.side_to_team.items():
            if t == team:
                other_side = "right" if side == "left" else "left"
                return self.side_to_team.get(other_side)
        return None


def save_team_checkpoint(team_name: str, agents: List, side: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    payload = []
    for ag in agents:
        if getattr(ag, "team", "").upper() == team_name.upper() and getattr(ag, "side", "left") == side:
            payload.append({
                "state_dict": ag.policy_net.state_dict(),
            })
    path = os.path.join(out_dir, f"{team_name.upper()}_{side}.pth")
    torch.save(payload, path)
    print(f"Checkpoint saved for team {team_name} side {side} -> {path}")


def load_team_checkpoint(team_name: str, agents: List, side: str, ckpt_dir: str):
    path = os.path.join(ckpt_dir, f"{team_name.upper()}_{side}.pth")
    if not os.path.exists(path):
        return
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint {path}: {e}")
        return
    idx = 0
    for ag in agents:
        if getattr(ag, "team", "").upper() != team_name.upper() or getattr(ag, "side", "left") != side:
            continue
        if idx < len(payload):
            ag.policy_net.load_state_dict(payload[idx]["state_dict"])
            ag.target_net.load_state_dict(payload[idx]["state_dict"])
            idx += 1
    print(f"Loaded checkpoint for team {team_name} side {side} from {path}")


def build_players(field: Field, team_formations: Dict[str, List], team_name: str, side: str) -> List[Dict[str, Any]]:
    tpl = []
    base_positions = team_formations.get(team_name, team_formations["A"])
    for role, (x, y) in base_positions:
        px = x if side == "left" else field.width - x
        tpl.append({"team": team_name, "role": role, "x": px, "y": y, "vx": 0.0, "vy": 0.0, "side": side,
                    "home_x": px, "home_y": y, "base_home_x": px, "base_home_y": y})
    return tpl


def parse_args():
    p = argparse.ArgumentParser(description="Headless trainer for Colab")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--duration-steps", type=int, default=1000)
    p.add_argument("--team-left", type=str, default="A")
    p.add_argument("--team-right", type=str, default="B")
    p.add_argument("--video-mode", type=str, default="none", choices=["none", "last", "interval", "all", "always"])
    p.add_argument("--video-interval", type=int, default=10)
    p.add_argument("--scenario-seed", type=int, default=0, help="0 = always default formation")
    p.add_argument("--scenario-random-pct", type=float, default=50.0, help="0-100 chance to use scenario vs default (ignored if seed=0)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--video-dir", type=str, default="videos")
    return p.parse_args()


def main():
    args = parse_args()
    FPS = args.fps
    DURATION_STEPS = args.duration_steps
    CHECKPOINT_DIR = args.checkpoint_dir
    LOG_DIR = args.log_dir
    LOG_STEP_DIR = os.path.join(LOG_DIR, "steps")
    LOG_SUMMARY_DIR = os.path.join(LOG_DIR, "summary")
    VIDEO_DIR = args.video_dir
    os.makedirs(LOG_STEP_DIR, exist_ok=True)
    os.makedirs(LOG_SUMMARY_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

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

    stats_tracker = StatsTracker(field, FPS, args.team_left, args.team_right)

    for ep in range(args.episodes):
        base_players = build_players(field, team_formations, args.team_left, "left") + build_players(field, team_formations, args.team_right, "right")
        scenario_info = sample_episode_scenario(
            base_players,
            field,
            args.team_left,
            args.team_right,
            ep,
            args.scenario_seed,
            args.scenario_random_pct if args.scenario_seed != 0 else 0.0,
        )
        scenario_name = scenario_info.get("name", "default")
        scenario_template = scenario_info.get("template", "default")
        scenario_variant = scenario_info.get("variant", 0)
        players = scenario_info["players"]

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

        load_team_checkpoint(args.team_left, agents, "left", CHECKPOINT_DIR)
        load_team_checkpoint(args.team_right, agents, "right", CHECKPOINT_DIR)

        ball = Ball(field_width=field.width, field_height=field.height)
        if "ball" in scenario_info:
            bx, by = scenario_info["ball"]
            ball.x = bx
            ball.y = by
        recorder = Recorder()
        save_video = False
        if args.video_mode == "interval":
            save_video = ((ep + 1) % max(1, args.video_interval) == 0)
        elif args.video_mode == "last":
            save_video = (ep == args.episodes - 1)
        elif args.video_mode in ("all", "always"):
            save_video = True

        renderer = None
        exporter = None
        if save_video:
            renderer = PygameRenderer(width=field.width, height=field.height, scale=10, show_horizontal_zones=False, show_vertical_zones=False)
            video_path = os.path.join(VIDEO_DIR, f"simulation_ep{ep+1}.mp4")
            exporter = VideoExporter(video_path, fps=FPS)

        simulator = Simulator(agents, players, ball, field, recorder, fps=FPS)

        old_state = simulator.snapshot()
        total_rewards = [0.0 for _ in agents]
        step_rows: List[Dict[str, Any]] = []
        stats_tracker.start_episode(players, scenario_name=scenario_name, scenario_variant=scenario_variant)

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

            ball_ctrl = new_state.get("ball_controller")
            ball_ctrl_team = ""
            if ball_ctrl is not None and isinstance(ball_ctrl, int) and 0 <= ball_ctrl < len(players):
                ball_ctrl_team = players[ball_ctrl].get("team", "")
            step_rows.append({
                "episode": ep + 1,
                "step": step + 1,
                "scenario": scenario_name,
                "scenario_template": scenario_template,
                "scenario_variant": scenario_variant,
                "ball_x": new_state.get("ball", {}).get("x", 0.0),
                "ball_y": new_state.get("ball", {}).get("y", 0.0),
                "ball_controller": ball_ctrl if ball_ctrl is not None else -1,
                "ball_controller_team": ball_ctrl_team,
                "last_goal": simulator.last_goal or "",
                "reward_p0": total_rewards[0],
                "actions": json.dumps(actions),
            })
            old_state = new_state

            if renderer and exporter:
                frame = renderer.render(new_state)
                exporter.add_frame(frame)

            if done:
                break

        if exporter and renderer:
            exporter.export()
            renderer.quit()
            print(f"[Ep {ep+1}] video saved to {exporter.path}")

        ep_rows, window_rows_100, window_rows_1000 = stats_tracker.finish_episode(ep+1)
        print(f"[Ep {ep+1}] rewards (sum): {total_rewards}")
        for row in ep_rows:
            if row["category"] == "team":
                print(f"[Ep {ep+1}] Team {row['team']}: possession {row['possession_pct']:.1f}%, goals {row['goals_for']} kebobolan {row['goals_against']} pass acc {row['pass_accuracy']:.1f}%")

        step_path = os.path.join(LOG_STEP_DIR, f"episode_{ep+1}_steps.csv")
        with open(step_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "step", "scenario", "scenario_template", "scenario_variant",
                                                   "ball_x", "ball_y", "ball_controller", "ball_controller_team", "last_goal", "reward_p0", "actions"])
            writer.writeheader()
            writer.writerows(step_rows)

        summary_rows = ep_rows + window_rows_100 + window_rows_1000
        summary_path = os.path.join(LOG_SUMMARY_DIR, f"summary_episode_{ep+1}.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "category", "window", "episode", "entity", "team", "role",
                "scenario", "scenario_variant",
                "possession_pct", "dribbles", "tackles", "blocks",
                "passes", "passes_completed", "pass_accuracy",
                "shots_on", "shots_off", "goals", "clears", "saves",
                "goals_for", "goals_against",
            ])
            writer.writeheader()
            writer.writerows(summary_rows)

        save_team_checkpoint(args.team_left, agents, "left", CHECKPOINT_DIR)
        save_team_checkpoint(args.team_right, agents, "right", CHECKPOINT_DIR)
        if (ep + 1) % 100 == 0:
            save_team_checkpoint(f"{args.team_left}_ep{ep+1}", agents, "left", CHECKPOINT_DIR)
            save_team_checkpoint(f"{args.team_right}_ep{ep+1}", agents, "right", CHECKPOINT_DIR)

    print("Training selesai.")


if __name__ == "__main__":
    main()
