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
BLOCK_RADIUS = 1.2
BLOCK_REFLECT = 0.4
BALL_DRAG_PER_SEC = 0.45    # ~0.64 speed left after 1s if not touched
BALL_RESTITUTION = 0.55     # pantulan saat mengenai garis lapangan

class Simulator:
    def __init__(self, agents, players, ball, field, recorder, fps=1):
        self.agents = agents
        self.players = players
        self.ball = ball
        self.ball_controller = None          # store index or None
        self.field = field
        self.recorder = recorder
        self.last_goal = None
        self.last_goal_scorer = None
        self.last_touch = None
        self.out_of_bounds = False
        self.offside = False
        self.dt = 1.0 / fps
        self.step_count = 0
        self.state = []
        self.action = []

    def reset(self):
        self.last_goal = None
        self.last_goal_scorer = None
        self.out_of_bounds = False
        self.last_touch = None
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
            'last_goal_scorer': self.last_goal_scorer,
            'last_touch': self.last_touch,
            'out_of_bounds': self.out_of_bounds,
            'step': self.step_count,
            'field': {'width': getattr(self.field, 'width', 100.0), 'height': getattr(self.field, 'height', 75.0)}
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
                # gunakan langsung tanpa smoothing agar aksi DQN tidak ditimpa
                p['vx'], p['vy'] = tvx, tvy
                self._apply_role_constraints(p, dt)
                continue
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
            self._apply_role_constraints(p, dt)

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
            goal_side = self.field.is_goal(self.ball.x, self.ball.y) if hasattr(self.field, "is_goal") else None
            if goal_side:
                self.last_goal = goal_side
                self.last_goal_scorer = self.last_touch
                self.ball_controller = None
                if hasattr(self.ball, "vx"):
                    self.ball.vx = 0.0
                    self.ball.vy = 0.0
                return
        else:
            # integrasi bola bebas dengan drag dt-invariant dan pantulan garis
            vx = getattr(self.ball, 'vx', 0.0)
            vy = getattr(self.ball, 'vy', 0.0)
            self.ball.x += vx * dt
            self.ball.y += vy * dt
            drag = math.exp(-BALL_DRAG_PER_SEC * dt)
            if hasattr(self.ball, 'vx'):
                self.ball.vx *= drag
                self.ball.vy *= drag
            # pantulan dinding lapangan (kecuali lewat mulut gawang)
            radius = getattr(self.ball, 'radius', 0.11)
            # cek goal terlebih dahulu
            goal_side = self.field.is_goal(self.ball.x, self.ball.y) if hasattr(self.field, "is_goal") else None
            if goal_side:
                self.last_goal = goal_side
                self.last_goal_scorer = self.last_touch
                self.ball_controller = None
                if hasattr(self.ball, "vx"):
                    self.ball.vx = 0.0
                    self.ball.vy = 0.0
                return
            # pantul vertikal (atas/bawah)
            if self.ball.y < radius:
                self.ball.y = radius
                self.ball.vy = abs(self.ball.vy) * BALL_RESTITUTION
            elif self.ball.y > self.field.height - radius:
                self.ball.y = self.field.height - radius
                self.ball.vy = -abs(self.ball.vy) * BALL_RESTITUTION
            # pantul horizontal (kiri/kanan) jika bukan gawang
            if self.ball.x < radius:
                self.ball.x = radius
                self.ball.vx = abs(self.ball.vx) * BALL_RESTITUTION
            elif self.ball.x > self.field.width - radius:
                self.ball.x = self.field.width - radius
                self.ball.vx = -abs(self.ball.vx) * BALL_RESTITUTION
        # peluang blok bola oleh pemain lain (tanpa kontrol)
        if self.ball_controller is None:
            for i, p in enumerate(self.players):
                dx = self.ball.x - p['x']
                dy = self.ball.y - p['y']
                d2 = dx*dx + dy*dy
                if d2 < BLOCK_RADIUS*BLOCK_RADIUS:
                    # d20 roll sederhana: sukses blok bila roll/20 < 0.7
                    if random.random() < 0.7:
                        speed = math.hypot(self.ball.vx, self.ball.vy)
                        nx, ny = (dx / (math.sqrt(d2)+1e-6), dy / (math.sqrt(d2)+1e-6))
                        # refleksi sebagian ke arah berlawanan
                        self.ball.vx = -nx * speed * BLOCK_REFLECT
                        self.ball.vy = -ny * speed * BLOCK_REFLECT
                        # bola berhenti dekat pemain, kontrol peluang
                        if speed < 12.0 and random.random() < 0.5:
                            self.ball_controller = i

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
                ball_speed2 = (getattr(self.ball, 'vx', 0.0)**2 + getattr(self.ball, 'vy', 0.0)**2)
                if d2 < min_d2 and ball_speed2 < 36.0:
                    # d20 roll: peluang sukses kontrol menurun saat bola cepat
                    roll = random.random()  # 0-1; treat as roll/20
                    success_prob = max(0.2, 1.0 - (ball_speed2 / 400.0))  # cepat => lebih sulit
                    if roll < success_prob:
                        min_idx = i
                        min_d2 = d2
            if min_idx is not None:
                self.ball_controller = min_idx
                self.last_touch = min_idx

    def _post_ball_update(self):
        # tidak lagi clamp keras; pantulan ditangani di _update_ball
        pass

    def _clamp_players(self):
        for p in self.players:
            p['x'] = max(0, min(self.field.width, p['x']))
            p['y'] = max(0, min(self.field.height, p['y']))

    def _apply_role_constraints(self, player, dt):
        role = str(player.get('role', '')).lower()
        side = player.get('side', 'left')
        fw, fh = self.field.width, self.field.height
        home_x_base = player.get('base_home_x', player.get('home_x', fw / 2))
        home_y_base = player.get('base_home_y', player.get('home_y', fh / 2))
        home_x = home_x_base
        home_y = home_y_base
        dir_sign = 1 if side == 'left' else -1
        vx = player.get('vx', 0.0)
        vy = player.get('vy', 0.0)
        nx = player.get('x', 0.0) + vx * dt
        ny = player.get('y', 0.0) + vy * dt
        teammates = [pl for pl in self.players if pl is not player and pl.get('team') == player.get('team')]
        cover_count = 0
        for tm in teammates:
            tx = tm.get('x', 0.0)
            if (side == 'left' and tx < player.get('x', 0.0)) or (side == 'right' and tx > player.get('x', 0.0)):
                cover_count += 1

        # Goalkeeper: jaga area gawang, sweep hanya bila bola dekat dan risiko rendah.
        if 'goalkeeper' in role:
            x_min, x_max = (0.0, 18.0) if side == 'left' else (fw - 18.0, fw)
            y_min, y_max = fh * 0.2, fh * 0.8
            goal_x = 0.0 if side == 'left' else fw
            ball_d = math.hypot(self.ball.x - player.get('x', 0.0), self.ball.y - player.get('y', 0.0))
            ball_to_goal = abs(self.ball.x - goal_x)
            safe_to_sweep = (ball_to_goal < 25.0 and ball_d < 20.0)
            outward = vx * dir_sign > 0
            if outward and not safe_to_sweep:
                vx *= 0.2
            if nx < x_min or nx > x_max:
                vx = 0.0
            if ny < y_min or ny > y_max:
                vy = 0.0
            player['vx'], player['vy'] = vx, vy
            return

        is_defender = ('center back' in role) or ('fullback' in role)
        if is_defender:
            adv_line = fw * 0.6 if side == 'left' else fw * 0.4
            hold_line = fw * 0.52 if side == 'left' else fw * 0.48
            ball_side_ok = (self.ball.x <= adv_line) if side == 'left' else (self.ball.x >= adv_line)
            allow_press = ball_side_ok or cover_count >= 1
            outward = vx * dir_sign > 0
            if outward and not allow_press and ((side == 'left' and nx > hold_line) or (side == 'right' and nx < hold_line)):
                vx = 0.0
            max_y_delta = 18.0
            if ny > home_y + max_y_delta or ny < home_y - max_y_delta:
                vy = 0.0
            if 'fullback' in role:
                flank_band = 14.0
                if ny > home_y + flank_band or ny < home_y - flank_band:
                    vy = 0.0
                allow_overlap = ((self.ball.x > fw * 0.55 and side == 'left') or (self.ball.x < fw * 0.45 and side == 'right')) and cover_count >= 1
                overlap_cap = hold_line + 8.0 if side == 'left' else hold_line - 8.0
                if outward and not allow_overlap and ((side == 'left' and nx > overlap_cap) or (side == 'right' and nx < overlap_cap)):
                    vx = 0.0
            player['vx'], player['vy'] = vx, vy
            return

        if 'winger' in role:
            flank_band = 14.0
            if ny > home_y + flank_band or ny < home_y - flank_band:
                vy = 0.0
            # tahan terlalu dalam kecuali build dari belakang
            min_line = fw * 0.38 if side == 'left' else fw * 0.62
            allow_drop = (self.ball.x < fw * 0.4) if side == 'left' else (self.ball.x > fw * 0.6)
            retreating = vx * dir_sign < 0
            if retreating and not allow_drop and ((side == 'left' and nx < min_line) or (side == 'right' and nx > min_line)):
                vx = 0.0
            # dorong bila bola sudah maju
            if not retreating and ((side == 'left' and self.ball.x < fw * 0.45) or (side == 'right' and self.ball.x > fw * 0.55)):
                vx *= 0.6
            player['vx'], player['vy'] = vx, vy
            return

        if 'midfielder' in role:
            if 'wing' not in role:
                band_y = 12.0
                if ny > home_y + band_y or ny < home_y - band_y:
                    vy = 0.0
                floor_line = fw * 0.22 if side == 'left' else fw * 0.78
                if (side == 'left' and nx < floor_line) or (side == 'right' and nx > floor_line):
                    vx = 0.0
                # jangan over-commit jika tanpa cover
                press_line = fw * 0.7 if side == 'left' else fw * 0.3
                forward = vx * dir_sign > 0
                if forward and cover_count == 0 and ((side == 'left' and nx > press_line) or (side == 'right' and nx < press_line)):
                    vx = 0.0
                player['vx'], player['vy'] = vx, vy
                return

        if 'striker' in role:
            band_y = 18.0
            if ny > home_y + band_y or ny < home_y - band_y:
                vy = 0.0
            drop_floor = fw * 0.35 if side == 'left' else fw * 0.65
            hold_line = fw * 0.48 if side == 'left' else fw * 0.52
            retreating = vx * dir_sign < 0
            allow_drop = (self.ball.x < fw * 0.38) if side == 'left' else (self.ball.x > fw * 0.62)
            if retreating and not allow_drop and ((side == 'left' and nx < hold_line) or (side == 'right' and nx > hold_line)):
                vx *= 0.3
            if (side == 'left' and nx < drop_floor) or (side == 'right' and nx > drop_floor):
                vx = 0.0
            player['vx'], player['vy'] = vx, vy

    def apply_action(self, player, action, dt):
        if not isinstance(action, dict):
            return
        # Raw velocity override (used by DQN). This keeps desired v through smoothing step.
        if "move_vx" in action or "move_vy" in action:
            player['vx'] = float(action.get('move_vx', player.get('vx', 0.0)))
            player['vy'] = float(action.get('move_vy', player.get('vy', 0.0)))
            player['tx'] = player['x']
            player['ty'] = player['y']
            player['speed'] = math.hypot(player['vx'], player['vy'])
        # Kick handling when controller adalah pemain ini
        if "kick_power" in action and action.get("kick_power", 0.0) > 0.0:
            idx = self.players.index(player)
            if self.ball_controller == idx:
                dx, dy = action.get("kick_dir", (1.0, 0.0))
                mag = (dx*dx + dy*dy) ** 0.5
                if mag > 1e-6:
                    dx /= mag; dy /= mag
                power = float(action.get("kick_power", 0.0))
                speed = 25.0 * max(0.0, min(1.0, power))
                self.ball.vx = dx * speed
                self.ball.vy = dy * speed
                self.last_touch = idx
                self.ball_controller = None

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
def compute_agent_reward(simulator, agent, old_state, new_state, action):
    """
    Reward shaping sederhana untuk multi-agen DQN.
    """
    r = 0.0
    team = agent.team.upper()
    side = getattr(agent, "side", "left")
    idx = getattr(agent, "player_index", 0)
    fw = new_state.get('field', {}).get('width', 100.0) if new_state else 100.0
    fh = new_state.get('field', {}).get('height', 75.0) if new_state else 75.0

    def ctrl_team(ctrl, players):
        if isinstance(ctrl, dict):
            return ctrl.get("team", None)
        if isinstance(ctrl, int) and players and 0 <= ctrl < len(players):
            return players[ctrl].get("team", None)
        if isinstance(ctrl, str):
            return ctrl
        return None

    role = getattr(agent, "role_name", "").lower()
    is_gk = "goalkeeper" in role
    is_def = any(k in role for k in ["back", "fullback"])
    is_mid = "midfielder" in role and not is_def
    is_wing = "winger" in role
    is_striker = "striker" in role and not is_wing
    role_key = (
        "gk" if is_gk else
        "def" if is_def else
        "wing" if is_wing else
        "striker" if is_striker else
        "mid"
    )

    role_goal_scale = {"gk": 5.0, "def": 4.5, "mid": 3.4, "wing": 4.0, "striker": 4.2}
    team_gain_scale = {"gk": 1.6, "def": 1.5, "mid": 1.1, "wing": 1.2, "striker": 1.25}
    team_loss_scale = {"gk": 1.6, "def": 1.45, "mid": 1.1, "wing": 1.0, "striker": 1.0}
    player_gain_scale = {"gk": 1.2, "def": 1.1, "mid": 1.0, "wing": 1.05, "striker": 1.1}
    player_loss_scale = {"gk": 1.2, "def": 1.1, "mid": 1.0, "wing": 1.0, "striker": 1.0}
    progress_scale = {"gk": 0.35, "def": 0.55, "mid": 1.0, "wing": 1.3, "striker": 1.45}
    goal_prox_scale = {"gk": 0.6, "def": 0.75, "mid": 1.0, "wing": 1.25, "striker": 1.35}
    own_third_loss_scale = {"gk": 1.4, "def": 1.35, "mid": 1.1, "wing": 1.0, "striker": 1.0}

    own_goal_x = 0.0 if side == 'left' else fw
    own_goal_y = fh / 2.0

    def scale(table, default=1.0):
        return table.get(role_key, default)

    # goal reward/penalty (lebih besar, bobot per role)
    if simulator.last_goal:
        outcome = 0.0
        if simulator.last_goal == 'right':   # menyerang kanan
            outcome = 1.0 if side == 'left' else -1.0
        elif simulator.last_goal == 'left':  # menyerang kiri
            outcome = 1.0 if side == 'right' else -1.0
        if outcome != 0.0:
            r += outcome * scale(role_goal_scale, 3.5)

    # possession change
    old_ctrl = old_state.get('ball_controller') if old_state else None
    new_ctrl = new_state.get('ball_controller')
    old_ctrl_team = ctrl_team(old_ctrl, old_state.get('players') if old_state else [])
    new_ctrl_team = ctrl_team(new_ctrl, new_state.get('players'))
    if new_ctrl == idx and old_ctrl != idx:
        r += 0.25 * scale(player_gain_scale, 1.0)
    if old_ctrl == idx and new_ctrl != idx:
        r -= 0.25 * scale(player_loss_scale, 1.0)
    if new_ctrl_team == team and old_ctrl_team != team:
        bonus = 0.15 * scale(team_gain_scale, 1.0)
        r += bonus  # tim merebut bola
    if old_ctrl_team == team and new_ctrl_team not in (team, None):
        penalty = 0.2 * scale(team_loss_scale, 1.0)
        r -= penalty   # tim kehilangan bola
    if new_ctrl == idx:
        r += 0.03

    # progress bola menuju gawang lawan
    old_ball = old_state.get('ball') if old_state else None
    new_ball = new_state.get('ball')
    if old_ball and new_ball:
        dx = (new_ball['x'] - old_ball['x'])
        signed_dx = dx if side == 'left' else -dx
        progress_gain = 0.003 * signed_dx * scale(progress_scale, 1.0)
        if new_ctrl_team == team:
            progress_gain *= 2.2
        r += progress_gain

    # mendekati bola
    players_old = old_state.get('players') if old_state else None
    players_new = new_state.get('players')
    if players_old and players_new and 0 <= idx < len(players_old):
        po = players_old[idx]
        pn = players_new[idx]
        bo = old_ball if old_ball else {'x':0,'y':0}
        bn = new_ball if new_ball else {'x':0,'y':0}
        dist_old = ((po['x']-bo['x'])**2 + (po['y']-bo['y'])**2) ** 0.5
        dist_new = ((pn['x']-bn['x'])**2 + (pn['y']-bn['y'])**2) ** 0.5
        r += 0.01 * (dist_old - dist_new)

    # bola mendekati gawang lawan saat tim menguasai
    if new_ball:
        target_x = fw if side == 'left' else 0.0
        dist_to_goal = abs(target_x - new_ball['x'])
        goal_prox = max(0.0, 1.0 - dist_to_goal / max(1e-3, fw))
        if new_ctrl_team == team:
            prox_bonus = 0.05 * goal_prox * scale(goal_prox_scale, 1.0)
            r += prox_bonus

    # waktu
    r -= 0.0008

    # end conditions penalty/bonus
    if simulator.out_of_bounds:
        r -= 0.3  # penalti bola keluar
    if simulator.offside:
        r -= 0.2  # penalti offside

    # penalti kehilangan bola di sepertiga sendiri (lebih berat untuk GK/def)
    if old_ball and old_ctrl_team == team and new_ctrl_team not in (team, None):
        own_third = fw / 3.0
        if (side == 'left' and old_ball['x'] < own_third) or (side == 'right' and old_ball['x'] > (fw - own_third)):
            loss_pen = 0.3 * scale(own_third_loss_scale, 1.0)
            r -= loss_pen

    # Event-based heuristik per role
    if new_ball:
        goal_dist = math.hypot(new_ball['x'] - own_goal_x, new_ball['y'] - own_goal_y)
        own_third = fw / 3.0
        in_own_third = (side == 'left' and new_ball['x'] < own_third) or (side == 'right' and new_ball['x'] > (fw - own_third))
        # GK: tangkap/kuasai bola di sekitar gawang
        if is_gk and new_ctrl == idx and goal_dist < 12.0:
            r += 0.6
        # Def/GK: merebut bola di area sendiri
        if (is_gk or is_def) and new_ctrl == idx and old_ctrl_team not in (team, None) and in_own_third:
            r += 0.3
        # Penalty jika lawan kuasai bola di kotak dekat gawang
        if (is_gk or is_def) and new_ctrl_team not in (team, None) and goal_dist < 12.0:
            r -= 0.35
        # Clear dari area sendiri (kick power > 0)
        if (is_gk or is_def) and old_ctrl == idx and in_own_third:
            if isinstance(action, dict) and float(action.get("kick_power", 0.0) or 0.0) > 0.0:
                r += 0.22

    return r
