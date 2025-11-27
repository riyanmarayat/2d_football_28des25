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
        # Return a lightweight immutable view of the current simulation state
        return {
            'players': [dict(p) for p in self.players],  # shallow copy of player dicts
            'ball': (self.ball.x, self.ball.y),  # renderer expects a (x, y) tuple
            'field': {
                'width': self.field.width,
                'height': self.field.height,
            },
            'ball_controller': dict(self.ball_controller) if isinstance(self.ball_controller, dict) else None,
            'last_goal': self.last_goal,
            'out_of_bounds': self.out_of_bounds,
            'offside': self.offside,
            'step': self.step_count,
        }

    def step(self, dt=1.0):
        self.step_count += 1

        # inisialisasi pengontrol bola jika None: pilih pemain terdekat
        if self.ball_controller is None and self.players:
            closest = min(self.players, key=lambda p: math.hypot(self.ball.x - p['x'], self.ball.y - p['y']))
            if math.hypot(self.ball.x - closest['x'], self.ball.y - closest['y']) <= getattr(self.ball, 'control_radius', 2.0):
                self.ball_controller = closest

        for agent, player in zip(self.agents, self.players):
            state = self.get_observation(agent, player)
            action = agent.decide_action(state, self.ball_controller)
            self.apply_action(player, action, dt)

        # update bola (friksi/gerak)
        if hasattr(self.ball, 'update'):
            self.ball.update(dt=dt, field=self.field)

        goal = self.field.is_goal(self.ball.x, self.ball.y)
        if goal:
            self.last_goal = goal

        # bounds check sederhana
        if hasattr(self.field, 'in_bounds') and hasattr(self.ball, 'radius'):
            if not self.field.in_bounds(self.ball.x, self.ball.y, radius=self.ball.radius):
                self.out_of_bounds = True

    def apply_action(self, player, action, dt):
        # action contoh: {'type': 'move', 'target': (tx, ty), 'speed': 5.0}
        if not isinstance(action, dict):
            return
        atype = action.get('type')
        if atype == 'move':
            tx, ty = action.get('target', (player['x'], player['y']))
            speed = float(action.get('speed', 5.0))
            dx = tx - player['x']
            dy = ty - player['y']
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                ux, uy = dx / dist, dy / dist
                step = speed * dt
                if step >= dist:
                    player['x'], player['y'] = tx, ty
                else:
                    player['x'] += ux * step
                    player['y'] += uy * step
            # clamp ke ukuran lapangan
            player['x'] = max(0, min(self.field.width, player['x']))
            player['y'] = max(0, min(self.field.height, player['y']))
        elif atype == 'control':
            # ambil kontrol bola jika dekat
            dist = math.hypot(self.ball.x - player['x'], self.ball.y - player['y'])
            if dist <= getattr(self.ball, 'control_radius', 2.0):
                self.ball_controller = player
        elif atype == 'pass':
            # {'type':'pass', 'target':(tx,ty), 'power':10.0}
            if self.ball_controller is player and hasattr(self.ball, 'kick_towards'):
                tx, ty = action.get('target', (player['x'], player['y']))
                power = float(action.get('power', 10.0))
                self.ball.kick_towards(tx, ty, power)
                self.ball_controller = None
        elif atype == 'shoot':
            # {'type':'shoot', 'power':12.0}
            if self.ball_controller is player and hasattr(self.field, 'get_opponent_goal_center') and hasattr(self.ball, 'kick_towards'):
                gx, gy = self.field.get_opponent_goal_center(player['team'])
                power = float(action.get('power', 12.0))
                self.ball.kick_towards(gx, gy, power)
                self.ball_controller = None
        else:
            # no-op / aksi tidak dikenal
            pass

    def get_observation(self, agent, player):
        # use whichever your agent implements
        return agent.get_state(player, self.players, self.ball, self.field)
        # or:
        # return agent.get_observation(player, self.players, self.ball, self.field)

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
    # +1 for scoting (detect from simulator.last_goal)
    # +0.1 for a successfull pass (you'd need to check if bball.x moved toward teammate)
    # -0.5 if striker lost ball (old_state.has_ball and not new_state.has_ball
    r = 0.0
    if getattr(simulator, 'last_goal', None) == ('right' if striker.team.upper() == "A" else 'left'):
        r += 1.0
    if action['type'] == 'control':
        #crude control check: ball is now closer to player than old_state
        #implement your own logic here
        r += 0.01
    if action['type'] == 'pass':
        #crude pass-succes check: ball is now closer to target than old_state
        #implement your own logic here
        r += 0.1
    if old_state['ball_controller'] == 'me' and new_state['ball_controller'] != 'opponent':
        r -= 0.5
    return r