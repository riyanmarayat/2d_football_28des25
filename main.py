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
from agents.striker.qlearning_striker import QLearningStriker
FPS = 15
DURATION_STEPS = 1000

qpa0_striker = QLearningStriker(team="A")
players = []
#TEAM A (LEFT)
initial_positions_A = [
    # ('A', 'Goalkeeper', (5, 37.5)),
    # ('A', 'Right Fullback', (15, 60)),
    # ('A', 'Left Fullback', (15, 15)),
    # ('A', 'Center Back', (15, 30)),
    # ('A', 'Center Back', (15, 45)),
    # ('A', 'Defensive Midfielder', (25, 30)),
    # ('A', 'Defensive Midfielder', (25, 45)),
    # ('A', 'Attacking Midfielder', (35, 37.5)),
    # ('A', 'Right Winger', (35, 60)),
    # ('A', 'Left Winger', (35, 15)),
    # ('A', 'Striker', (45, 37.5))
    ('A', 'Striker', (5, 37.5)),
    ('A', 'Striker', (15, 60)),
    ('A', 'Striker', (15, 15)),
    # ('A', 'Striker', (15, 30)),
    # ('A', 'Striker', (15, 45)),
    # ('A', 'Striker', (25, 30)),
    # ('A', 'Striker', (25, 45)),
    # ('A', 'Striker', (35, 37.5)),
    # ('A', 'Striker', (35, 60)),
    # ('A', 'Striker', (35, 15)),
    # ('A', 'Striker', (45, 37.5))
]
for team, role, (x, y) in initial_positions_A:
    players.append({'team': team, 'role':role, 'x': x, 'y': y})
#TEAM B (RIGHT)
initial_positions_B = [
    # ('B', 'Goalkeeper', (95, 37.5)),
    # ('B', 'Center Back', (85, 50)),
    # ('B', 'Center Back', (85, 37.5)),
    # ('B', 'Center Back', (85, 25)),
    # ('B', 'Right Wingback', (75, 15)),
    # ('B', 'Left Wingback', (75, 60)),
    # ('B', 'Right Midfielder', (70, 20)),
    # ('B', 'Left Midfielder', (70, 55)),
    # ('B', 'Central Midfielder', (70, 37.5)),
    # ('B', 'Striker', (55, 45)),
    # ('B', 'Striker', (55, 30))
    # ('B', 'Striker', (95, 37.5)),
    # ('B', 'Striker', (85, 50)),
    # ('B', 'Striker', (85, 37.5)),
    # ('B', 'Striker', (85, 25)),
    # ('B', 'Striker', (75, 15)),
    # ('B', 'Striker', (75, 60)),
    # ('B', 'Striker', (70, 20)),
    # ('B', 'Striker', (70, 55)),
    ('B', 'Striker', (70, 37.5)),
    ('B', 'Striker', (55, 45)),
    ('B', 'Striker', (55, 30))
]
for team, role, (x, y) in initial_positions_B:
    players.append({'team': team, 'role':role, 'x': x, 'y': y})

ball = Ball(field_width=100, field_height=75)
field = Field(width=100, height=75)
recorder = Recorder()
renderer = PygameRenderer(width=100, height=75, scale=10, show_horizontal_zones=False, show_vertical_zones=False)
exporter = VideoExporter("simulation.mp4", fps=FPS)

#map each player dict to its agent instanc
agents = []
player_count = 0
for p in players:
    if player_count == 0:
        agents.append(qpa0_striker)
        player_count += 1
        continue
    player_count += 1
    team = p['team'].upper()
    role = p['role'].lower()
    if role ==  'goalkeeper':
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
#build sim
simulator = Simulator(agents, players, ball, field, recorder, fps=FPS)
old_state = None
total_reward = 0
# total_episode = 0
#main loop
current_time = 0.0
for step in range(DURATION_STEPS):
    dt = 1.0 / FPS
    if step == 500:
        pass
    simulator.step(dt=dt) #update logic and physuics
    # learning
    new_state = qpa0_striker.get_state(players[0], players, ball, field)
    if old_state is None:
        old_state = new_state
    # compute reward and update Q-Table
    reward = computer_striker_reward(simulator, qpa0_striker, old_state, new_state, qpa0_striker.last_action_dict)
    print(f"Step {step+1}/{DURATION_STEPS} reward: {reward} || {qpa0_striker.last_action_dict} || {qpa0_striker.ball_controller}")
    qpa0_striker.learn(reward, new_state)
    total_reward += reward
    old_state = new_state
    if simulator.last_goal:
        break
    elif simulator.out_of_bounds:
        print(f"Ball out of bounds at ({simulator.ball.x:.2f}, {simulator.ball.y:.2f})")
        break
    # elif simulator.offside:
    #     print("Offside! Ending this play")
    #     break

    frame = renderer.render(simulator.snapshot())
    exporter.add_frame(frame)
    current_time += dt
    own_ball = simulator.ball_controller
    if own_ball is None:
        print("No player is controlling the ball")
    else:
        print(f"player {own_ball['team']} {own_ball['role']} is controlling the ball at ({own_ball['x']:.2f}, {own_ball['y']:.2f})")
    # # print(f"Step {step+1}/{DURATION_STEPS}|| striker 4 x:{agents[3].get_state(players[3], players, ball, field)}")
    # print(f"Step {step + 1}/{DURATION_STEPS}|| striker 4 action:{agents[3].decide_action(agents[3].get_state(players[3], players, ball, field), simulator.ball_controller)}")
    # # print(f"Step {step+1}/{DURATION_STEPS}|| striker 5 x:{agents[4].get_state(players[4], players, ball, field)}")
    print(f"Step {step + 1}/{DURATION_STEPS}|| striker 5 action:{agents[4].decide_action(agents[4].get_state(players[4], players, ball, field), simulator.ball_controller)}")
    # # print(f"Step {step+1}/{DURATION_STEPS}|| striker 6 x:{agents[5].get_state(players[5], players, ball, field)}")
    # print(f"Step {step + 1}/{DURATION_STEPS}|| striker 6 action:{agents[5].decide_action(agents[5].get_state(players[5], players, ball, field), simulator.ball_controller)}")
    # # print(f"Step {step+1}/{DURATION_STEPS} RW: {players[8]}")


#finalize
exporter.export()
renderer.quit()
print("Simulation save to simulation.mp4")
print(f"Episode reward: {total_reward}")