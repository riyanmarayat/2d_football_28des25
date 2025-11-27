import math
import random
from agents.common_rules import decide_action, evaluate_context
from agents.striker.striker import StrikerAgent
from agents.midfielder.midfielder import midfielder_rules
from agents.defender.defender import defender_rules
from agents.goalkeeper.goalkeeper import goalkeeper_rules
from core.physics import resolve_player_collisions

DRIBBLE_OFFSET = 0.8
CONTROL_RADIUS = 1.2
RELEASE_RADIUS = 1.8
ACCEL_FACTOR = 6.0
MAX_SPEED = 6.0
SUBSTEPS = 2

class Simulator:
    def __init__(self, agents, players, ball, field, recorder, fps=1):
        self.agents = agents
        self.players = players
        self.ball = ball
        self.ball_controller = None          # store index or None
        self.field = field
        self.recorder = recorder
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
        self.ball_controller = None
        for p in self.players:
            # players are dict; ensure basic fields reset if needed
            p['vx'] = 0.0
            p['vy'] = 0.0
        if hasattr(self.ball, 'reset'):
            self.ball.reset()
        else:
            self.ball.x = self.field.width / 2
            self.ball.y = self.field.height / 2
            self.ball.vx = 0.0
            self.ball.vy = 0.0

    def snapshot(self):
        return {
            'players': [dict(p) for p in self.players],
            'ball': {'x': self.ball.x, 'y': self.ball.y, 'vx': getattr(self.ball, 'vx', 0.0), 'vy': getattr(self.ball, 'vy', 0.0)},
            'ball_pos': (self.ball.x, self.ball.y),
            'ball_controller': self.ball_controller,
            'last_goal': self.last_goal,
            'out_of_bounds': self.out_of_bounds,
            'step': self.step_count
        }

    def step(self, dt: float):
        sub_dt = dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            self._agents_decide(sub_dt)
            self._update_player_motion(sub_dt)
            self._update_ball(sub_dt)
            self._handle_possession()
            self._post_ball_update()
        self._record_snapshot()

    def _agents_decide(self, dt):
        for p, agent in zip(self.players, self.agents):
            # jika agent punya desired_velocity gunakan itu
            if hasattr(agent, 'desired_velocity'):
                tvx, tvy = agent.desired_velocity(p, self.ball, self.field)
            else:
                # fallback: gunakan target (tx, ty) atau kejar bola
                if 'tx' in p and 'ty' in p:
                    dx = p['tx'] - p['x']
                    dy = p['ty'] - p['y']
                else:
                    dx = self.ball.x - p['x']
                    dy = self.ball.y - p['y']
                dist = (dx*dx + dy*dy) ** 0.5
                desired_speed = p.get('speed', MAX_SPEED)
                if dist > 1e-4:
                    tvx = dx / dist * desired_speed
                    tvy = dy / dist * desired_speed
                else:
                    tvx = 0.0
                    tvy = 0.0
            # smoothing
            p['vx'] += (tvx - p['vx']) * ACCEL_FACTOR * dt
            p['vy'] += (tvy - p['vy']) * ACCEL_FACTOR * dt

    def _update_player_motion(self, dt):
        for p in self.players:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
        resolve_player_collisions(self.players)
        self._clamp_players()

    def _update_ball(self, dt):
        if self.ball_controller is not None:
            pc = self.players[self.ball_controller]
            speed = (pc['vx']**2 + pc['vy']**2)**0.5
            if speed > 0.01:
                nx = pc['vx'] / speed
                ny = pc['vy'] / speed
            else:
                nx, ny = 1.0, 0.0
            self.ball.x = pc['x'] + nx * DRIBBLE_OFFSET
            self.ball.y = pc['y'] + ny * DRIBBLE_OFFSET
            self.ball.vx = pc['vx']
            self.ball.vy = pc['vy']
        else:
            self.ball.x += getattr(self.ball, 'vx', 0.0) * dt
            self.ball.y += getattr(self.ball, 'vy', 0.0) * dt
            if hasattr(self.ball, 'vx'):
                self.ball.vx *= 0.985
                self.ball.vy *= 0.985

    def _handle_possession(self):
        if self.ball_controller is not None:
            pc = self.players[self.ball_controller]
            dx = self.ball.x - pc['x']
            dy = self.ball.y - pc['y']
            if dx*dx + dy*dy > RELEASE_RADIUS*RELEASE_RADIUS:
                self.ball_controller = None
        if self.ball_controller is None:
            min_idx = None
            min_d2 = CONTROL_RADIUS * CONTROL_RADIUS
            for i, p in enumerate(self.players):
                dx = self.ball.x - p['x']
                dy = self.ball.y - p['y']
                d2 = dx*dx + dy*dy
                if d2 < min_d2 and (getattr(self.ball, 'vx', 0.0)**2 + getattr(self.ball, 'vy', 0.0)**2) < 9.0:
                    min_idx = i
                    min_d2 = d2
            if min_idx is not None:
                self.ball_controller = min_idx

    def _post_ball_update(self):
        if self.ball.x < 0: self.ball.x = 0
        if self.ball.x > self.field.width: self.ball.x = self.field.width
        if self.ball.y < 0: self.ball.y = 0
        if self.ball.y > self.field.height: self.ball.y = self.field.height

    def _clamp_players(self):
        for p in self.players:
            p['x'] = max(0, min(self.field.width, p['x']))
            p['y'] = max(0, min(self.field.height, p['y']))

    def apply_action(self, player, action, dt):
        if not isinstance(action, dict):
            return
        t = action.get('type')
        if t == 'move':
            tx, ty = action.get('target', (player['x'], player['y']))
            speed = float(action.get('speed', MAX_SPEED))
            player['tx'] = tx
            player['ty'] = ty
            player['speed'] = speed
        elif t == 'control':
            # kontrol bola akan di-handle _handle_possession via radius
            pass
        elif t in ('pass', 'shoot'):
            idx = self.players.index(player)
            if self.ball_controller == idx and hasattr(self.ball, 'kick_towards'):
                if t == 'pass':
                    tx, ty = action.get('target', (player['x'], player['y']))
                    power = float(action.get('power', 10.0))
                    self.ball.kick_towards(tx, ty, power)
                else:
                    if hasattr(self.field, 'get_opponent_goal_center'):
                        gx, gy = self.field.get_opponent_goal_center(player['team'])
                    else:
                        gx, gy = self.field.width, self.field.height / 2
                    power = float(action.get('power', 12.0))
                    self.ball.kick_towards(gx, gy, power)
                self.ball_controller = None
        # clamp stays
        player['x'] = max(0, min(self.field.width, player['x']))
        player['y'] = max(0, min(self.field.height, player['y']))

    def _record_snapshot(self):
        snap = self.snapshot()
        # fleksibel terhadap API recorder
        if hasattr(self.recorder, 'add'):
            self.recorder.add(snap)
        elif hasattr(self.recorder, 'record'):
            self.recorder.record(snap)
        elif hasattr(self.recorder, 'append'):
            self.recorder.append(snap)

    # tetap (reward dll) ...
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