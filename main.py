import time
from core.ball import Ball
from core.field import Field
from core.recorder import Recorder
from core.simulator import Simulator, computer_striker_reward
from render.pygame_renderer import PygameRenderer
from render.video_exporter import VideoExporter
from agents.roles import (RightWingerAgent, LeftWingerAgent, AttackingMidfielderAgent,
                          RightMidfielderAgent, LeftMidfielderAgent, CentralMidfielderAgent, RightWingbackAgent,
                          LeftWingbackAgent, DefensiveMidfielderAgent, RightFullbackAgent, LeftFullbackAgent,
                          CenterBackAgent, GoalkeeperAgent)
from agents.striker.striker import StrikerAgent
from agents.striker.dqn_striker import DQNStriker

FPS = 15
DURATION_STEPS = 1000

# Inisialisasi DQN Striker
qpa0_striker = DQNStriker(team="A", seed=0)  # gunakan seed untuk reproducible

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

players = []
# TEAM A (LEFT)
initial_positions_A = [
    ('A', 'Striker', (5, 37.5)),
    ('A', 'Striker', (15, 60)),
    ('A', 'Striker', (15, 15)),
]
for team, role, (x, y) in initial_positions_A:
    players.append({'team': team, 'role': role, 'x': x, 'y': y, 'vx': 0.0, 'vy': 0.0})

# TEAM B (RIGHT)
initial_positions_B = [
    ('B', 'Striker', (70, 37.5)),
    ('B', 'Striker', (55, 45)),
    ('B', 'Striker', (55, 30))
]
for team, role, (x, y) in initial_positions_B:
    players.append({'team': team, 'role': role, 'x': x, 'y': y, 'vx': 0.0, 'vy': 0.0})

ball = Ball(field_width=100, field_height=75)
field = Field(width=100, height=75)
recorder = Recorder()
renderer = PygameRenderer(width=100, height=75, scale=10, show_horizontal_zones=False, show_vertical_zones=False)
exporter = VideoExporter("simulation.mp4", fps=FPS)

# Map each player dict to its agent instance
agents = []
player_count = 0
for p in players:
    p['tx'] = p['x']
    p['ty'] = p['y']
    p['speed'] = 0.0
    if player_count == 0:
        # Player[0] dikontrol oleh DQNStriker
        agents.append(qpa0_striker)
        player_count += 1
        continue
    player_count += 1
    team = p['team'].upper()
    role = p['role'].lower()
    if role == 'goalkeeper':
        agents.append(GoalkeeperAgent(team=team))
    elif role == 'center back':
        agents.append(CenterBackAgent(team=team))
    elif role == 'right fullback':
        agents.append(RightFullbackAgent(team=team))
    elif role == 'left fullback':
        agents.append(LeftFullbackAgent(team=team))
    elif role == 'defensive midfielder':
        agents.append(DefensiveMidfielderAgent(team=team))
    elif role == 'right wingback':
        agents.append(RightWingbackAgent(team=team))
    elif role == 'left wingback':
        agents.append(LeftWingbackAgent(team=team))
    elif role == 'central midfielder':
        agents.append(CentralMidfielderAgent(team=team))
    elif role == 'right midfielder':
        agents.append(RightMidfielderAgent(team=team))
    elif role == 'left midfielder':
        agents.append(LeftMidfielderAgent(team=team))
    elif role == 'attacking midfielder':
        agents.append(AttackingMidfielderAgent(team=team))
    elif role == 'right winger':
        agents.append(RightWingerAgent(team=team))
    elif role == 'left winger':
        agents.append(LeftWingerAgent(team=team))
    elif role == 'striker':
        agents.append(StrikerAgent(team=team))
    else:
        raise ValueError(f"Unknown role: {role}")

# Build simulator
simulator = Simulator(agents, players, ball, field, recorder, fps=FPS)

old_state = simulator.snapshot()
total_reward = 0

# Initialize last_state for DQN (extract once from snapshot)
_ = qpa0_striker.extract_features(old_state)

for step in range(DURATION_STEPS):
    dt = 1.0 / FPS

    # Pilih aksi DQN dari snapshot lama (state diekstraksi internal)
    action_idx = qpa0_striker.select_action(old_state)
    action_dict = qpa0_striker.action_index_to_dict(action_idx)

    # Terapkan aksi ke player[0]
    pre_x, pre_y = players[0]["x"], players[0]["y"]
    pre_vx, pre_vy = players[0].get("vx", 0.0), players[0].get("vy", 0.0)
    simulator.apply_action(players[0], action_dict, dt)

    # Jika tidak ada perubahan setelah apply_action, gunakan fallback
    no_move = (
        players[0]["x"] == pre_x and players[0]["y"] == pre_y and
        players[0].get("vx", 0.0) == pre_vx and players[0].get("vy", 0.0) == pre_vy
    )
    if no_move:
        apply_action_compat(simulator, players[0], action_dict, dt)

    # Step simulator
    simulator.step(dt=dt)

    # Snapshot baru
    new_state = simulator.snapshot()
    reward = computer_striker_reward(simulator, qpa0_striker, old_state, new_state, action_dict)
    done = bool(simulator.last_goal or simulator.out_of_bounds)

    # Learn
    if qpa0_striker.last_action_idx is not None:
        qpa0_striker.learn(reward, new_state, done)

    print(f"Step {step+1}/{DURATION_STEPS} reward: {reward} || {action_dict} || pos=({players[0]['x']:.2f},{players[0]['y']:.2f})")
    total_reward += reward
    old_state = new_state

    frame = renderer.render(new_state)
    exporter.add_frame(frame)

    if done:
        break

# finalize
exporter.export()
renderer.quit()
print(f"Simulation save to {exporter.path}")
print(f"Episode reward: {total_reward}")