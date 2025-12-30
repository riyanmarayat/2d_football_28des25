import random
from typing import List, Dict, Any, Tuple

# Helper utilities to shape player positions for scenarios.


def _shuffle_indices(indices: List[int], rng: random.Random) -> List[int]:
    idxs = list(indices)
    rng.shuffle(idxs)
    return idxs


def _move_group(players: List[Dict[str, Any]], indices: List[int], cx: float, cy: float, spread_x: float,
                spread_y: float, rng: random.Random):
    """Place a group around (cx, cy) with rectangular spread."""
    for idx in indices:
        if idx < 0 or idx >= len(players):
            continue
        px = cx + rng.uniform(-spread_x, spread_x)
        py = cy + rng.uniform(-spread_y, spread_y)
        players[idx]["x"] = px
        players[idx]["y"] = py


def _indices_by_role(players: List[Dict[str, Any]], team: str, role_keywords: List[str]) -> List[int]:
    idxs = []
    for i, p in enumerate(players):
        if p.get("team") != team:
            continue
        role = str(p.get("role", "")).lower()
        if any(k in role for k in role_keywords):
            idxs.append(i)
    return idxs


def _set_ball_position(ball_center: Tuple[float, float], rng: random.Random, jitter: float = 0.0) -> Tuple[float, float]:
    if jitter <= 0:
        return ball_center
    return (ball_center[0] + rng.uniform(-jitter, jitter),
            ball_center[1] + rng.uniform(-jitter, jitter))


def _template_kickoff_high_press(players, field, rng, team_left, team_right):
    # Ball at center; right team presses high, left team slightly compact.
    ball_pos = _set_ball_position((field.width / 2, field.height / 2), rng, jitter=1.0)
    left_defs = _indices_by_role(players, team_left, ["back", "goalkeeper"])
    _move_group(players, left_defs, field.width * 0.18, field.height / 2, 4, 18, rng)
    right_pressers = _indices_by_role(players, team_right, ["striker", "wing", "midfielder"])
    _move_group(players, right_pressers, field.width * 0.42, field.height / 2, 6, 22, rng)
    return ball_pos


def _template_counter_attack_left(players, field, rng, team_left, team_right):
    # Left just regained ball near halfway, runners ahead.
    ball_pos = _set_ball_position((field.width * 0.45, field.height * 0.55), rng, jitter=2.5)
    left_attack = _indices_by_role(players, team_left, ["striker", "wing"])
    _move_group(players, left_attack, field.width * 0.65, field.height * 0.55, 7, 15, rng)
    right_defs = _indices_by_role(players, team_right, ["back", "goalkeeper"])
    _move_group(players, right_defs, field.width * 0.8, field.height / 2, 4, 18, rng)
    return ball_pos


def _template_low_block_right(players, field, rng, team_left, team_right):
    # Right sits deep, left circulates ball in final third.
    ball_pos = _set_ball_position((field.width * 0.7, field.height * 0.45), rng, jitter=2.0)
    right_block = _indices_by_role(players, team_right, ["back", "midfielder", "goalkeeper"])
    _move_group(players, right_block, field.width * 0.87, field.height / 2, 5, 16, rng)
    left_attack = _indices_by_role(players, team_left, ["striker", "wing", "midfielder"])
    _move_group(players, left_attack, field.width * 0.72, field.height / 2, 6, 22, rng)
    return ball_pos


def _template_corner_left_attack(players, field, rng, team_left, team_right):
    # Left takes attacking corner.
    corner_y = rng.uniform(field.height * 0.2, field.height * 0.8)
    ball_pos = (0.5, corner_y)
    left_box = _indices_by_role(players, team_left, ["striker", "wing", "midfielder"])
    _move_group(players, left_box, field.width * 0.05, corner_y, 6, 10, rng)
    right_box = _indices_by_role(players, team_right, ["back", "goalkeeper"])
    _move_group(players, right_box, field.width * 0.04, corner_y, 4, 14, rng)
    return ball_pos


def _template_goal_kick_right_build(players, field, rng, team_left, team_right):
    # Right has goal kick, trying to build short.
    ball_pos = (field.width - 1.0, field.height / 2 + rng.uniform(-4, 4))
    right_defs = _indices_by_role(players, team_right, ["back", "goalkeeper"])
    _move_group(players, right_defs, field.width * 0.92, field.height / 2, 3, 18, rng)
    right_mids = _indices_by_role(players, team_right, ["midfielder"])
    _move_group(players, right_mids, field.width * 0.78, field.height / 2, 6, 20, rng)
    left_press = _indices_by_role(players, team_left, ["striker", "wing"])
    _move_group(players, left_press, field.width * 0.7, field.height / 2, 5, 18, rng)
    return ball_pos


def _template_wide_switch(players, field, rng, team_left, team_right):
    # Ball on left flank, right team shifting, left preparing switch.
    flank_y = rng.uniform(field.height * 0.2, field.height * 0.35)
    ball_pos = _set_ball_position((field.width * 0.55, flank_y), rng, jitter=1.5)
    left_support = _indices_by_role(players, team_left, ["wing", "midfielder"])
    _move_group(players, left_support, field.width * 0.6, flank_y, 5, 8, rng)
    right_shift = _indices_by_role(players, team_right, ["midfielder", "back"])
    _move_group(players, right_shift, field.width * 0.7, flank_y, 5, 12, rng)
    return ball_pos


def _template_penalty_box_scramble(players, field, rng, team_left, team_right):
    # Chaos in box: ball loose near right goal.
    ball_pos = _set_ball_position((field.width * 0.92, field.height / 2), rng, jitter=2.2)
    scrum_left = _indices_by_role(players, team_left, ["striker", "midfielder", "wing"])
    _move_group(players, scrum_left, field.width * 0.9, field.height / 2, 6, 10, rng)
    scrum_right = _indices_by_role(players, team_right, ["back", "goalkeeper", "midfielder"])
    _move_group(players, scrum_right, field.width * 0.94, field.height / 2, 4, 12, rng)
    return ball_pos


def _template_fast_break_right(players, field, rng, team_left, team_right):
    # Right counterattacking with runners.
    ball_pos = _set_ball_position((field.width * 0.35, field.height * 0.48), rng, jitter=2.0)
    right_runners = _indices_by_role(players, team_right, ["wing", "striker"])
    _move_group(players, right_runners, field.width * 0.55, field.height * 0.5, 8, 16, rng)
    left_recover = _indices_by_role(players, team_left, ["back", "midfielder"])
    _move_group(players, left_recover, field.width * 0.42, field.height / 2, 6, 18, rng)
    return ball_pos


def _template_mid_press_trap(players, field, rng, team_left, team_right):
    # Midfield trap: left tries to force right inside.
    ball_pos = _set_ball_position((field.width * 0.52, field.height * 0.5), rng, jitter=1.0)
    left_shape = _indices_by_role(players, team_left, ["midfielder", "wing"])
    _move_group(players, left_shape, field.width * 0.55, field.height / 2, 5, 18, rng)
    right_shape = _indices_by_role(players, team_right, ["midfielder", "back"])
    _move_group(players, right_shape, field.width * 0.5, field.height / 2, 6, 18, rng)
    return ball_pos


TEMPLATES = [
    {"name": "kickoff_high_press", "desc": "Kickoff with pressing shape from right side", "fn": _template_kickoff_high_press},
    {"name": "counter_attack_left", "desc": "Left regains and breaks quickly", "fn": _template_counter_attack_left},
    {"name": "low_block_right", "desc": "Right defends deep block, left in final third", "fn": _template_low_block_right},
    {"name": "corner_left_attack", "desc": "Left attacking corner situation", "fn": _template_corner_left_attack},
    {"name": "goal_kick_right_build", "desc": "Right goal-kick short build-up vs press", "fn": _template_goal_kick_right_build},
    {"name": "wide_switch_left", "desc": "Left attacking on left flank preparing switch", "fn": _template_wide_switch},
    {"name": "penalty_box_scramble", "desc": "Loose ball in right penalty area", "fn": _template_penalty_box_scramble},
    {"name": "fast_break_right", "desc": "Right counter from mid third with runners", "fn": _template_fast_break_right},
    {"name": "midfield_press_trap", "desc": "Both teams set mid-block pressing trap", "fn": _template_mid_press_trap},
]


def sample_episode_scenario(
    base_players: List[Dict[str, Any]],
    field,
    team_left: str,
    team_right: str,
    episode_idx: int,
    scenario_seed: int,
    randomness_pct: float,
) -> Dict[str, Any]:
    """
    Decide whether to use default or randomized scenario for this episode.
    If scenario_seed == 0 -> always default (existing formation).
    randomness_pct in [0,100]: chance to use randomized scenario instead of default.
    """
    if scenario_seed == 0:
        return {
            "players": base_players,
            "ball": (field.width / 2, field.height / 2),
            "name": "default",
            "template": "default",
            "variant": 0,
        }
    rng = random.Random(scenario_seed + episode_idx)
    use_random = rng.random() < max(0.0, min(100.0, randomness_pct)) / 100.0
    if not use_random:
        return {
            "players": base_players,
            "ball": (field.width / 2, field.height / 2),
            "name": "default",
            "template": "default",
            "variant": episode_idx,
        }

    players = [dict(p) for p in base_players]
    template = rng.choice(TEMPLATES)
    ball_pos = template["fn"](players, field, rng, team_left, team_right)
    return {
        "players": players,
        "ball": ball_pos,
        "name": f"{template['name']}",
        "template": template["name"],
        "variant": scenario_seed + episode_idx,
    }
