import time
import os
import torch
from typing import List, Dict, Any
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
NUM_EPISODES = int(os.environ.get("NUM_EPISODES", 1))
CHECKPOINT_DIR = "checkpoints"

def save_team_checkpoint(team_name: str, agents: List):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    payload = []
    for ag in agents:
        if getattr(ag, "team", "").upper() == team_name.upper():
            payload.append({
                "state_dict": ag.policy_net.state_dict(),
            })
    path = os.path.join(CHECKPOINT_DIR, f"{team_name.upper()}.pth")
    torch.save(payload, path)
    print(f"Checkpoint saved for team {team_name} -> {path}")

def load_team_checkpoint(team_name: str, agents: List):
    path = os.path.join(CHECKPOINT_DIR, f"{team_name.upper()}.pth")
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
        if getattr(ag, "team", "").upper() != team_name.upper():
            continue
        if idx < len(payload):
            ag.policy_net.load_state_dict(payload[idx]["state_dict"])
            ag.target_net.load_state_dict(payload[idx]["state_dict"])
            idx += 1
    print(f"Loaded checkpoint for team {team_name} from {path}")

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

def build_players(team_name: str, side: str) -> List[Dict[str, Any]]:
    tpl = []
    base_positions = team_formations.get(team_name, team_formations["A"])
    for role, (x, y) in base_positions:
        px = x if side == "left" else field.width - x
        tpl.append({"team": team_name, "role": role, "x": px, "y": y, "vx": 0.0, "vy": 0.0, "side": side})
    return tpl

for ep in range(NUM_EPISODES):
    # rebuild players and agents per episode to keep indices/side aligned
    players = build_players(team_left, "left") + build_players(team_right, "right")
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

    # Load checkpoints for both teams if available
    load_team_checkpoint(team_left, agents)
    load_team_checkpoint(team_right, agents)

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
    print(f"Episode {ep+1} saved to {exporter.path}")
    print(f"Episode rewards (sum): {total_rewards}")
    # simpan log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Episode log saved to {log_path}")

    # Save checkpoints per team
    save_team_checkpoint(team_left, agents)
    save_team_checkpoint(team_right, agents)
    if (ep + 1) % 100 == 0:
        save_team_checkpoint(f"{team_left}_ep{ep+1}", agents)
        save_team_checkpoint(f"{team_right}_ep{ep+1}", agents)
