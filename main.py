import time
from core.ball import Ball
from core.field import Field
from core.recorder import Recorder
from core.simulator import Simulator, computer_striker_reward
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
# TEAM A (LEFT) 4-4-2
initial_positions_A = [
    ('A', 'Goalkeeper', (5, 37.5)),
    ('A', 'Left Fullback', (18, 20)),
    ('A', 'Center Back', (18, 32)),
    ('A', 'Center Back', (18, 45)),
    ('A', 'Right Fullback', (18, 57)),
    ('A', 'Left Midfielder', (35, 20)),
    ('A', 'Central Midfielder', (35, 32)),
    ('A', 'Central Midfielder', (35, 45)),
    ('A', 'Right Midfielder', (35, 57)),
    ('A', 'Striker', (55, 32)),
    ('A', 'Striker', (55, 45)),
]
for team, role, (x, y) in initial_positions_A:
    players.append({'team': team, 'role': role, 'x': x, 'y': y, 'vx': 0.0, 'vy': 0.0})

# TEAM B (RIGHT) 4-3-3
initial_positions_B = [
    ('B', 'Goalkeeper', (95, 37.5)),
    ('B', 'Right Fullback', (82, 20)),
    ('B', 'Center Back', (82, 32)),
    ('B', 'Center Back', (82, 45)),
    ('B', 'Left Fullback', (82, 57)),
    ('B', 'Central Midfielder', (65, 25)),
    ('B', 'Central Midfielder', (65, 37.5)),
    ('B', 'Central Midfielder', (65, 50)),
    ('B', 'Right Winger', (55, 20)),
    ('B', 'Striker', (55, 37.5)),
    ('B', 'Left Winger', (55, 55)),
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
for idx, p in enumerate(players):
    role = p['role'].lower()
    team = p['team']
    if role == 'goalkeeper':
        agents.append(DQNGoalkeeperAgent(team, player_index=idx, seed=idx))
    elif role == 'center back':
        agents.append(DQNCenterBackAgent(team, player_index=idx, seed=idx))
    elif role == 'right fullback':
        agents.append(DQNRightFullbackAgent(team, player_index=idx, seed=idx))
    elif role == 'left fullback':
        agents.append(DQNLeftFullbackAgent(team, player_index=idx, seed=idx))
    elif role == 'central midfielder':
        agents.append(DQNCentralMidfielderAgent(team, player_index=idx, seed=idx))
    elif role == 'right midfielder':
        agents.append(DQNRightMidfielderAgent(team, player_index=idx, seed=idx))
    elif role == 'left midfielder':
        agents.append(DQNLeftMidfielderAgent(team, player_index=idx, seed=idx))
    elif role == 'right winger':
        agents.append(DQNRightWingerAgent(team, player_index=idx, seed=idx))
    elif role == 'left winger':
        agents.append(DQNLeftWingerAgent(team, player_index=idx, seed=idx))
    elif role == 'striker':
        agents.append(DQNStriker(team, player_index=idx, seed=idx))  # atau buat kelas sendiri

# Build simulator
simulator = Simulator(agents, players, ball, field, recorder, fps=FPS)

old_state = simulator.snapshot()
total_rewards = [0.0 for _ in agents]

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

    # Step simulator
    simulator.step(dt=dt)

    # Snapshot baru
    new_state = simulator.snapshot()
    done = bool(simulator.last_goal or simulator.out_of_bounds)

    # Learn untuk tiap agen
    for i, agent in enumerate(agents):
        reward = computer_striker_reward(simulator, agent, old_state, new_state, actions[i])
        if agent.last_action_idx is not None:
            agent.learn(reward, new_state, done)
        total_rewards[i] += reward

    # log sederhana untuk agen pertama
    print(f"Step {step+1}/{DURATION_STEPS} reward_p0: {total_rewards[0]:.3f} pos=({players[0]['x']:.2f},{players[0]['y']:.2f})")
    old_state = new_state

    frame = renderer.render(new_state)
    exporter.add_frame(frame)

    if done:
        break

# finalize
exporter.export()
renderer.quit()
print(f"Simulation save to {exporter.path}")
print(f"Episode rewards (sum): {total_rewards}")
