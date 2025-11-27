import math
import random
from agents.common_rules import decide_action, evaluate_context
from agents.striker.striker import StrikerAgent
from agents.midfielder.midfielder import midfielder_rules
from agents.defender.defender import defender_rules
from agents.goalkeeper.goalkeeper import goalkeeper_rules

class Simulator:
    def __init__(self, agents, players, ball, field, recorder, fps=1):
        self.agents = agents #store logical agent isntances
        self.players = players #store player state data (list of dicts wiht, x, y, role, team)
        self.ball = ball #physics object for ball
        self.ball_controller = None #controller for ball physics
        self.field = field #field geometry and goal logic
        self.recorder = recorder #recorder for logging events
        self.last_goal = None
        self.out_of_bounds = False
        self.offside = False
        self.dt = 1.0 / fps
        self.step_count = 0
        self.state = []
        self.action = []

    def reset(self):
        self.last_goal = None
        self.out_of_bounds = False
        for player in self.players:
            player.reset()
        self.ball.reset()

    def snapshot(self):
        return {
            'players': [dict(p) for p in self.players],
            'ball': (self.ball.x, self.ball.y),
            'ball_controller': dict(self.ball_controller) if isinstance(self.ball_controller, dict) else None,
            'last_goal': self.last_goal,
            'out_of_bounds': self.out_of_bounds,
            'step': self.step_count
        }

    def step(self, dt=1.0):
        self.step_count += 1
        # auto assign ball controller if none
        if self.ball_controller is None and self.players:
            closest = min(self.players, key=lambda p: math.hypot(self.ball.x - p['x'], self.ball.y - p['y']))
            if math.hypot(self.ball.x - closest['x'], self.ball.y - closest['y']) <= getattr(self.ball, 'control_radius', 2.0):
                self.ball_controller = closest

        for agent, player in zip(self.agents, self.players):
            state = self.get_observation(agent, player)
            action = agent.decide_action(state, self.ball_controller)
            self.apply_action(player, action, dt)

        if hasattr(self.ball, 'update'):
            self.ball.update(dt=dt, field=self.field)

        goal = self.field.is_goal(self.ball.x, self.ball.y)
        if goal:
            self.last_goal = goal

        if hasattr(self.field, 'in_bounds') and hasattr(self.ball, 'radius'):
            if not self.field.in_bounds(self.ball.x, self.ball.y, radius=self.ball.radius):
                self.out_of_bounds = True

    def apply_action(self, player, action, dt):
        if not isinstance(action, dict):
            return
        t = action.get('type')
        if t == 'move':
            tx, ty = action.get('target', (player['x'], player['y']))
            speed = float(action.get('speed', 5.0))
            dx, dy = tx - player['x'], ty - player['y']
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                step = min(speed * dt, dist)
                player['x'] += dx / dist * step
                player['y'] += dy / dist * step
            player['x'] = max(0, min(self.field.width, player['x']))
            player['y'] = max(0, min(self.field.height, player['y']))
        elif t == 'control':
            dist = math.hypot(self.ball.x - player['x'], self.ball.y - player['y'])
            if dist <= getattr(self.ball, 'control_radius', 2.0):
                self.ball_controller = player
        elif t == 'pass':
            if self.ball_controller is player and hasattr(self.ball, 'kick_towards'):
                tx, ty = action.get('target', (player['x'], player['y']))
                power = float(action.get('power', 10.0))
                self.ball.kick_towards(tx, ty, power)
                self.ball_controller = None
        elif t == 'shoot':
            if self.ball_controller is player and hasattr(self.field, 'get_opponent_goal_center') and hasattr(self.ball, 'kick_towards'):
                gx, gy = self.field.get_opponent_goal_center(player['team'])
                power = float(action.get('power', 12.0))
                self.ball.kick_towards(gx, gy, power)
                self.ball_controller = None
        # else noop

    def get_observation(self, agent, player):
        # adapt to actual agent API
        if hasattr(agent, 'get_state'):
            return agent.get_state(player, self.players, self.ball, self.field)
        return agent.get_observation(player, self.players, self.ball, self.field)

    def get_observation_by_name(self, agent_name):
        for player in self.players:
            if player.name == agent_name:
                return self.get_observation(player)
        return None

    def apply_action_by_name(self, agent_name, action, dt=1.0):
        for player in self.players:
            if player.name == agent_name:
                self.apply_action(player, action, dt)
                break

    def get_reward(self, agent_name):
        for player in self.players:
            if player.name == agent_name:
                if self.last_goal:
                    if (self.last_goal == "right" and player.side == "left") or \
                            (self.last_goal == "left" and player.side == "right"):
                        return 1.0
                return 0.0
        return 0.0

    def check_terminated(self):
        return self.last_goal is not None

    def render(self):
        self.field.render(self.players, self.ball)

    def close(self):
        pass


def computer_striker_reward(simulator, striker, old_state, new_state, action):
    """
    Compute reward safely even if old_state is None.
    Expected new_state format can be adjusted; here we rely on simulator and action only.
    """
    r = 0.0
    # goal reward
    if simulator.last_goal:
        # if team A scores when attacking right goal
        scored_side = simulator.last_goal  # 'left' or 'right'
        # assume team A attacks right, team B attacks left
        if striker.team.upper() == 'A' and scored_side == 'right':
            r += 1.0
        elif striker.team.upper() == 'B' and scored_side == 'left':
            r += 1.0
    # ball control reward
    if action.get('type') == 'control':
        r += 0.05
    # pass reward (simple)
    if action.get('type') == 'pass':
        r += 0.1
    # maintain possession (needs old_state present)
    if old_state is not None and isinstance(old_state, dict):
        if old_state.get('ball_controller') == 'me' and new_state.get('ball_controller') == 'me':
            r += 0.02
        if old_state.get('ball_controller') == 'me' and new_state.get('ball_controller') not in ('me', None):
            r -= 0.3  # lost possession
    return r