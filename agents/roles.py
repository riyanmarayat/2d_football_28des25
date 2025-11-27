from agents.base_agent import FootballAgent
# from main import field


# from core.field import Field
#
# field = Field(100, 75)
# ----------------Striker & Attackers-------------------
class StrikerAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Striker")

    def get_state(self, self_player, players, ball, field):
        """
        Gather striker-specific features:
        - self_pos, ball, pos, goal_pos
        - dist_to_ball, dist_to_goal
        - nearest_opponents_dist, has_ball
        - best_teammate_pos, off_ball_run_pos
        :param self_player:
        :param players:
        :param ball:
        :param field:
        :return:
        """
        #ball controller
        if self.ball_controller is None:
            ball_controller = None
        elif self.ball_controller == self_player:
            ball_controller = 'me'
        elif self.ball_controller['team'] == self.team:
            ball_controller = 'teammate'
        else:
            ball_controller = 'opponent'

        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        #opponent goal center
        goal = field.right_goal if self.team.upper() == "A" else field.left_goal
        gx = goal['x']
        gy = (goal['y_top'] + goal['y_bottom']) / 2

        #distances
        dist_to_ball = ((sx - bx)**2 + (sy - by)**2)**0.5
        dist_to_goal = ((sx - gx)**2 + (sy - gy)**2)**0.5

        #closest opponent
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opponents_dist = min(
            ((sx - op['x'])**2 + (sy - op['y'])**2)**0.5 for op in opponents
        ) if opponents else float('inf')

        #best passing option: teammate with most open space
        teammates = [p for p in players if p['team'] == self.team and p is not self_player]
        def space_score(tm):
            #how far is this teammate from the nearest opponent
            dists = [((tm['x'] - op['x'])**2 + (tm['y'] - op['y'])**2)**0.5 for op in opponents]
            return min(dists) if dists else float('inf')
        best_teammate = None
        for tm in teammates:
            if best_teammate is None or space_score(tm) < space_score(best_teammate):
                best_teammate = tm
        best_teammate_pos = (best_teammate['x'], best_teammate['y'])

        #off-ball run position (10 units toward goal)
        direction = 1 if self.team.upper() == "A" else -1
        off_run_x = sx + direction * 10
        off_ball_run_pos = (off_run_x, sy)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'goal_pos': (gx, gy),
            'dist_to_ball': dist_to_ball,
            'dist_to_goal': dist_to_goal,
            'nearest_opponents_dist': nearest_opponents_dist,
            'ball_controller': ball_controller,
            'best_teammate_pos': best_teammate_pos,
            'off_ball_run_pos': off_ball_run_pos,
            'run_with_ball_pos': (sx + direction * 10, sy)
        }

    def decide_action(self, state, ball_controller=None):
        """
        -if in possession and clear path & close enough: shoot.
        -if in possession but pressured: pass
        -otherwise, chase the ball or make an off-ball run
        :param state:
        :return:
        """
        #control the ball first
        self.ball_controller = ball_controller
        if state['ball_controller'] is None:
            if state['dist_to_ball'] < 1:
                return {'type': 'control', 'target': state['ball_pos']}
            else:
                return {'type': 'move', 'target': state['ball_pos']}
        elif state['ball_controller'] == 'me':
            return {'type': 'run_with_ball', 'target': state['run_with_ball_pos']}
        else:
            return {'type': 'move', 'target': state['ball_pos']}
        # #with ball
        # if state['ball_controller'] == 'me':
        #     if state['dist_to_goal'] < 25 and state['nearest_opponents_dist'] > 8:
        #         return {'type': 'shoot', 'target': state['goal_pos']}
        #     return {'type': 'pass', 'target': state['best_teammate_pos']}
        # #without ball
        # if state['dist_to_ball'] < 20:
        #     return {'type': 'move', 'target': state['ball_pos']}
        # return {'type': 'move', 'target': state['off_ball_run_pos']}

class RightWingerAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Right Winger")
        self.field = None

    def get_state(self, self_player, players, ball, field):
        """

        :param self_player:
        :param players:
        :param ball:
        :param field:
        :return:
        """
        if field is not None:
            self.field = field
        # current posistions
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        #discante metrics
        dist_to_ball = ((sx - bx)**2 + (sy - by)**2)**0.5

        #sideline: x = field.width (right side)
        if self.team.upper() == "A":
            dist_to_sideline = abs(self.field.height - sx)
        else:
            dist_to_sideline = abs(0 - sx)

        #opponent pressure
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opponent_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # opponent goal center
        goal = field.right_goal if self.team.upper() == "A" else field.left_goal
        gx = goal['x']
        gy = (goal['y_top'] + goal['y_bottom']) / 2
        dist_to_goal = ((sx - gx)**2 + (sy - gy)**2)**0.5

        #crossing targets at near and far post
        if self.team.upper() == "A":
            near_post = (gx, gy+3)
            far_post = (gx, gy-3)
        else:
            near_post = (gx, gy-3)
            far_post = (gx, gy+3)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'dist_to_sideline': dist_to_sideline,
            'nearest_opponent_dist': nearest_opponent_dist,
            'dist_to_goal': dist_to_goal,
            'near_post': near_post,
            'far_post': far_post
        }

    def decide_action(self, state, ball_controller=None):
        #if we have the ball...
        if state['ball_controller'] == 'me':
            #in crossing range: >20 unit from goal and close to sideline
            if state['dist_to_goal'] < 40 and state['dist_to_sideline'] < 10:
                #choose the post with more space
                return {
                    'type': 'cross',
                    'target': state['near_post'] if state['nearest_opponent_dist'] > 15 else state['far_post']
                }
            #if unpressured, dribble toward goal
            if state['nearest_opponent_dist'] > 8:
                return {'type': 'move', 'target': state['goal_pos'] if 'goal_pos' in state else state['ball_pos']}
            #otherwise, keep the ball and wait
            return {'type': 'move', 'target': state['self_pos']}

        #if the ball is loose nearby, chase it
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}

        #defaylt: stay wide at 80% of pitch, align with ball
        if self.team.upper() == "A":
            return {'type': 'move', 'target': (state['ball_pos'][0], self.field.height * 0.8)}
        else:
            return {'type': 'move', 'target': (state['ball_pos'][0], self.field.height * 0.2)}

class LeftWingerAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Left Winger")
        self.field = None

    def get_state(self, self_player, players, ball, field):
        """

        :param self_player:
        :param players:
        :param ball:
        :param field:
        :return:
        """
        if field is not None:
            self.field = field
        # current posistions
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        #discante metrics
        dist_to_ball = ((sx - bx)**2 + (sy - by)**2)**0.5

        #sideline: x = field.width (left side)
        if self.team.upper() == "A":
            dist_to_sideline = abs(0 - sx)
        else:
            dist_to_sideline = abs(self.field.height - sx)

        #opponent pressure
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opponent_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # opponent goal center
        goal = field.right_goal if self.team.upper() == "A" else field.left_goal
        gx = goal['x']
        gy = (goal['y_top'] + goal['y_bottom']) / 2
        dist_to_goal = ((sx - gx)**2 + (sy - gy)**2)**0.5

        #crossing targets at near and far post
        if self.team.upper() == "A":
            near_post = (gx, gy-3)
            far_post = (gx, gy+3)
        else:
            near_post = (gx, gy+3)
            far_post = (gx, gy-3)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'dist_to_sideline': dist_to_sideline,
            'nearest_opponent_dist': nearest_opponent_dist,
            'dist_to_goal': dist_to_goal,
            'near_post': near_post,
            'far_post': far_post
        }

    def decide_action(self, state, ball_controller=None):
        #if we have the ball...
        if state['ball_controller'] == 'me':
            #in crossing range: >20 unit from goal and close to sideline
            if state['dist_to_goal'] < 40 and state['dist_to_sideline'] < 10:
                #choose the post with more space
                return {
                    'type': 'cross',
                    'target': state['near_post'] if state['nearest_opponent_dist'] > 15 else state['far_post']
                }
            #if unpressured, dribble toward goal
            if state['nearest_opponent_dist'] > 8:
                return {'type': 'move', 'target': state['goal_pos'] if 'goal_pos' in state else state['ball_pos']}
            #otherwise, keep the ball and wait
            return {'type': 'move', 'target': state['self_pos']}

        #if the ball is loose nearby, chase it
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}

        #defaylt: stay wide at 80% of pitch, align with ball
        if self.team.upper() == "A":
            return {'type': 'move', 'target': (state['ball_pos'][0], self.field.height * 0.2)}
        else:
            return {'type': 'move', 'target': (state['ball_pos'][0], self.field.height * 0.8)}

# ----------------Midfielders-------------------
class AttackingMidfielderAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Attacking Midfielder")

    def get_state(self, self_player, players, ball, field):
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        #find the striker's position
        strikers = [p for p in players if p['team']==self.team and p['role'].lower()=='striker']
        if strikers is None:
         strikers = self_player

        #distances
        dist_to_ball = ((sx - bx)**2 + (sy - by)**2)**0.5
        dist_to_striker = [((sx - striker['x'])**2 + (sy - striker['y'])**2)**0.5 for striker in strikers]

        #nearest opponent to ball (for pressure)
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opponent_dist = min(
            ((bx - op['x'])**2 + (by - op['y'])**2)**0.5 for op in opponents
        ) if opponents else float('inf')

        #support position: halfway beetwen striker and ball
        if len(strikers) > 1:
            striker = min(strikers, key=lambda pos: ((pos['x']-bx)**2 + (pos['y']-by)**2))
        else:
            striker = strikers[0]
        support_x = (striker['x'] + bx)/2
        support_y = (striker['y'] + by)/2

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'striker_pos': (striker['x'], striker['y']),
            'dist_to_ball': dist_to_ball,
            'dist_to_striker': dist_to_striker,
            'ball_controller': self.ball_controller,
            'nearest_opponent_dist': nearest_opponent_dist,
            'support_pos': (support_x, support_y)
        }

    def decide_action(self, state, ball_controller=None):
        #if in possession
        if state['ball_controller'] == 'me':
            #if unpressured and within 25 units of goal, dribble toward goal
            if state['nearest_opponent_dist'] > 8:
                #aim at space just ahead
                if self.team.upper() == "A":
                    return {'type': 'move', 'target': (state['self_pos'][0] + 10, state['self_pos'][1])}
                else:
                    return {'type': 'move', 'target': (state['self_pos'][0] - 10, state['self_pos'][1])}
            #othewrwise, play quick pass to striker if there free
            if state['dist_to_striker'][0] < 30 and state['nearest_opponent_dist'] > 5:
                return {'type': 'pass', 'target': state['striker_pos']}
            #if too much presure, drop back to support position
            return {'type': 'move', 'target': state['support_pos']}

        #without the ball: of it's loose nearby, chase it
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}
        #otherwise, move to support position
        return {'type': 'move', 'target': state['support_pos']}

class CentralMidfielderAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Central Midfielder")
        self.field = None

    def get_state(self, self_player, players, ball, field):
        if field is not None:
            self.field = field
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Distances and possession
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Determine half: attacking or defending
        attacking_half = bx > self.field.width / 2 if self.team.upper() == 'A' else bx < self.field.width / 2

        # Nearest teammate to pass (excluding self)
        teammates = [p for p in players if p['team'] == self.team and p is not self_player]

        def teammate_score(tm):
            # combine distance and openness
            dx, dy = tm['x'] - sx, tm['y'] - sy
            d = (dx ** 2 + dy ** 2) ** 0.5
            return -d

        best_tm = max(teammates, key=teammate_score) if teammates else self_player
        best_teammate_pos = (best_tm['x'], best_tm['y'])

        # Defensive fallback position (center of own half)
        def_x = self.field.width * 0.25 if self.team.upper() == 'A' else self.field.width * 0.75
        def_pos = (def_x, self.field.height / 2)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'attacking_half': attacking_half,
            'best_teammate_pos': best_teammate_pos,
            'def_pos': def_pos
        }

    def decide_action(self, state, ball_controller=None):
        # In possession
        if state['ball_controller'] == 'me':
            # If in attacking half, push forward
            if state['attacking_half']:
                if self.team.upper() == "A":
                    return {'type': 'move', 'target': (state['self_pos'][0] + 8, state['self_pos'][1])}
                else:
                    return {'type': 'move', 'target': (state['self_pos'][0] - 8, state['self_pos'][1])}
            # Else pass to best teammate
            return {'type': 'pass', 'target': state['best_teammate_pos']}

        # Without ball: if nearby, chase
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}
        # Otherwise, hold defensive position
        return {'type': 'move', 'target': state['def_pos']}

class DefensiveMidfielderAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Defensive Midfielder")

    def get_state(self, self_player, players, ball, field):
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Distance to ball and possession status
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Center of own goal for defensive positioning
        goal = field.left_goal if self.team.upper() == 'A' else field.right_goal
        gy = (goal['y_top'] + goal['y_bottom']) / 2
        goal_center = (goal['x'], gy)

        # Nearest opponent to intercept
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opp_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # Interception point between ball and goal
        intercept_x = (bx + goal_center[0]) / 2
        intercept_y = (by + goal_center[1]) / 2

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'goal_center': goal_center,
            'nearest_opp_dist': nearest_opp_dist,
            'intercept_pos': (intercept_x, intercept_y)
        }

    def decide_action(self, state, ball_controller=None):
        # If in possession close to own third, clear or pass
        if state['ball_controller']=='me' and state['dist_to_ball'] < 5:
            # If under pressure, pass to safe teammate
            if state['nearest_opp_dist'] < 8:
                return {'type': 'pass', 'target': state['goal_center']}
            # Otherwise, dribble into interception point
            return {'type': 'move', 'target': state['intercept_pos']}

        # If ball is loose nearby, move to intercept
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}

        # Default: hold interception position
        return {'type': 'move', 'target': state['intercept_pos']}


class RightMidfielderAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Right Midfielder")

    def get_state(self, self_player, players, ball, field):
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Distance to ball and possession
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Zone pressure: nearest opponent
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opp_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # Passing lanes: best central teammate
        teammates = [p for p in players if p['team'] == self.team and p is not self_player]

        def lane_score(tm):
            # prioritize forward positions with space
            dx, dy = tm['x'] - sx, abs(tm['y'] - sy)
            return -dy + dx * 0.5

        best_tm = max(teammates, key=lane_score) if teammates else self_player
        best_teammate_pos = (best_tm['x'], best_tm['y'])

        # Support run: 10 units upfield on right flank
        support_pos = (sx + (1 if self.team.upper() == 'A' else -1) * 10, sy)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'nearest_opp_dist': nearest_opp_dist,
            'best_teammate_pos': best_teammate_pos,
            'support_pos': support_pos
        }

    def decide_action(self, state, ball_controller=None):
        # With ball
        if state['ball_controller'] == 'me':
            # If unpressured and in attacking zone, drive ball upfield
            if state['nearest_opp_dist'] > 8:
                return {'type': 'move', 'target': state['support_pos']}
            # Otherwise pass to central support
            return {'type': 'pass', 'target': state['best_teammate_pos']}

        # Without ball: if loose ball is nearby, chase it
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}
        # Default: support upfield on right side
        return {'type': 'move', 'target': state['support_pos']}

class LeftMidfielderAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Left Midfielder")

    def get_state(self, self_player, players, ball, field):
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Distance to ball and possession
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Zone pressure: nearest opponent
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opp_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # Passing lanes: best central teammate
        teammates = [p for p in players if p['team'] == self.team and p is not self_player]

        def lane_score(tm):
            # prioritize forward positions with space
            dx, dy = tm['x'] - sx, abs(tm['y'] - sy)
            return -dy + dx * 0.5

        best_tm = max(teammates, key=lane_score) if teammates else self_player
        best_teammate_pos = (best_tm['x'], best_tm['y'])

        # Support run: 10 units upfield on left side
        direction = 1 if self.team.upper() == 'A' else -1
        support_pos = (sx + direction * 10, sy)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'nearest_opp_dist': nearest_opp_dist,
            'best_teammate_pos': best_teammate_pos,
            'support_pos': support_pos
        }


    def decide_action(self, state, ball_controller=None):
        # In possession
        if state['ball_controller'] == 'me':
            # If unpressured, drive upfield
            if state['nearest_opp_dist'] > 8:
                return {'type': 'move', 'target': state['support_pos']}
            # Otherwise, pass to best teammate
            return {'type': 'pass', 'target': state['best_teammate_pos']}

        # Without ball: chase if nearby
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}
        # Default: maintain support position
        return {'type': 'move', 'target': state['support_pos']}

# ----------------Defenders & Wingbacks-------------------
class RightWingbackAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Right Wingback")
        self.field = None

    def get_state(self, self_player, players, ball, field):
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Basic distances
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Distance to right sideline (x = field.width)
        if self.team.upper() == 'A':
            dist_to_sideline = abs(field.width - sx)
        else:
            dist_to_sideline = abs(0 - sx)

        # Opponent pressure
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opp_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # Support run upfield along wing
        direction = 1 if self.team.upper() == 'A' else -1
        support_pos = (sx + direction * 8, sy)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'dist_to_sideline': dist_to_sideline,
            'nearest_opp_dist': nearest_opp_dist,
            'support_pos': support_pos
        }


    def decide_action(self, state, ball_controller=None):
        # If in possession...
        if state['ball_controller'] == 'me':
            # If in attacking zone and unpressured, drive forward
            if state['nearest_opp_dist'] > 8 and state['dist_to_sideline'] < 15:
                return {'type': 'move', 'target': state['support_pos']}
            # If pressured, pass back or inward
            return {'type': 'pass', 'target': state['support_pos']}

        # If ball is loose nearby, chase it
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}

        # Default: stay on the wing in supporting position
        return {'type': 'move', 'target': state['support_pos']}

class LeftWingbackAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Left Wingback")

    def get_state(self, self_player, players, ball, field):
        # Extract positions
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Distance to ball and possession flag
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Distance to left sideline (x = 0)
        if self.team.upper() == 'A':
            dist_to_sideline = abs(0 - sx)
        else:
            dist_to_sideline = abs(field.width - sx)

        # Nearest opponent pressure
        opponents = [p for p in players if p['team'] != self.team]
        nearest_opp_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # Support run upfield along wing (mirror of right side)
        direction = 1 if self.team.upper() == 'A' else -1
        support_pos = (sx + direction * 8, sy)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'dist_to_sideline': dist_to_sideline,
            'nearest_opp_dist': nearest_opp_dist,
            'support_pos': support_pos
        }

    def decide_action(self, state, ball_controller=None):
        # If in possession on wing...
        if state['ball_controller'] == 'me':
            # If unpressured and near sideline, advance upfield
            if state['nearest_opp_dist'] > 8 and state['dist_to_sideline'] < 15:
                return {'type': 'move', 'target': state['support_pos']}
            # If pressured, pass inward or backwards to safety
            return {'type': 'pass', 'target': state['support_pos']}

        # If loose ball is within reach, chase it
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}

        # Default: maintain width and support position on the wing
        return {'type': 'move', 'target': state['support_pos']}

class RightFullbackAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Right Fullback")
        self.field = None

    def get_state(self, self_player, players, ball, field):
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Distance to ball and possession
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Defensive zone x-coordinate (30% into field)
        zone_x = field.width * (0.3 if self.team.upper() == 'A' else 0.7)

        # Opponent pressure: nearest striker
        opponents = [p for p in players if p['team'] != self.team]
        nearest_striker_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # Interception point between ball and defensive zone
        intercept_pos = ((bx + zone_x) / 2, by)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'nearest_striker_dist': nearest_striker_dist,
            'intercept_pos': intercept_pos
        }

    def decide_action(self, state, ball_controller=None):
        # Clear ball when in possession near own zone
        if state['ball_controller']=='me' and state['dist_to_ball'] < 5:
            return {'type': 'shoot', 'target': state['intercept_pos']}

        # Move to intercept loose ball
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}

        # Hold defensive line: stay on intercept position
        return {'type': 'move', 'target': state['intercept_pos']}

class LeftFullbackAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Left Fullback")

    def get_state(self, self_player, players, ball, field):
        sx, sy = self_player['x'], self_player['y']
        bx, by = ball.x, ball.y

        # Distance to ball and possession
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Defensive zone x-coordinate (30% into field from left)
        zone_x = field.width * (0.7 if self.team.upper() == 'B' else 0.3)

        # Opponent pressure: nearest striker
        opponents = [p for p in players if p['team'] != self.team]
        nearest_striker_dist = min(
            ((sx - op['x']) ** 2 + (sy - op['y']) ** 2) ** 0.5 for op in opponents
        ) if opponents else float('inf')

        # Interception point between ball and defensive zone
        intercept_pos = ((bx + zone_x) / 2, by)

        return {
            'self_pos': (sx, sy),
            'ball_pos': (bx, by),
            'dist_to_ball': dist_to_ball,
            'ball_controller': self.ball_controller,
            'nearest_striker_dist': nearest_striker_dist,
            'intercept_pos': intercept_pos
        }

    def decide_action(self, state, ball_controller=None):
        # Clear ball when in possession near own zone
        if state['ball_controller'] == 'me' and state['dist_to_ball'] < 5:
            return {'type': 'shoot', 'target': state['intercept_pos']}

        # Move to intercept loose ball
        if state['dist_to_ball'] < 20:
            return {'type': 'move', 'target': state['ball_pos']}

        # Hold defensive line: stay on intercept position
        return {'type': 'move', 'target': state['intercept_pos']}

class CenterBackAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Center Back")
        self.ball = None

    def get_state(self, self_player, players, ball, field):
        # Identify two most advanced opponents
        opponents = [p for p in players if p['team'] != self.team]
        strikers = sorted(opponents, key=lambda p: p['x'], reverse=(self.team.upper() == 'A'))[:2]
        return {
            'ball_pos': (ball.x, ball.y),
            'striker_positions': [(p['x'], p['y']) for p in strikers]
        }

    def decide_action(self, state, ball_controller=None):
        bx, by = state['ball_pos']
        # Mark the closest striker
        target = min(state['striker_positions'], key=lambda pos: ((pos[0] - bx) ** 2 + (pos[1] - by) ** 2))
        return {'type': 'move', 'target': target}

# ----------------Goalkeepers-------------------
class GoalkeeperAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Goalkeeper")

    def get_state(self, self_player, players, ball, field):
        """
                State includes:
                - ball_pos: current ball coordinates
                - goal_center: center of own goal
                - dist_to_ball: distance from keeper to ball
                - is_in_box: whether ball is inside penalty area
                - nearest_threat_pos: position of nearest opponent in box
                """
        # Ball and keeper positions
        bx, by = ball.x, ball.y
        sx, sy = self_player['x'], self_player['y']

        # Compute own goal center
        goal = field.left_goal if self.team.upper() == 'A' else field.right_goal
        gy = (goal['y_top'] + goal['y_bottom']) / 2
        gx = goal['x']

        # Distances
        dist_to_ball = ((sx - bx) ** 2 + (sy - by) ** 2) ** 0.5

        # Penalty area bounds
        x_min = field.width * 0.0 if self.team.upper() == 'A' else field.width * 0.8
        x_max = field.width * 0.2 if self.team.upper() == 'A' else field.width
        y_min = (field.height * 0.5) - (field.height * 0.2)
        y_max = (field.height * 0.5) + (field.height * 0.2)
        is_in_box = x_min <= bx <= x_max and y_min <= by <= y_max

        # Nearest opponent threat inside box
        threats = [p for p in players if
                   p['team'] != self.team and x_min <= p['x'] <= x_max and y_min <= p['y'] <= y_max]
        if threats:
            nearest_threat_pos = min(
                ((p['x'] - bx) ** 2 + (p['y'] - by) ** 2, (p['x'], p['y']))[1] for p in threats
            )
        else:
            nearest_threat_pos = None

        return {
            'ball_pos': (bx, by),
            'keeper_pos': (sx, sy),
            'goal_center': (gx, gy),
            'dist_to_ball': dist_to_ball,
            'is_in_box': is_in_box,
            'nearest_threat_pos': nearest_threat_pos
        }


    def decide_action(self, state, ball_controller=None):
        """
                - If ball inside box and close: move to intercept.
                - If ball in box but far: hold goal line.
                - If ball outside box: stay centered on goal line.
                - Distribute: if intercept and free throw option, pass to nearest teammate.
                """
        bx, by = state['ball_pos']
        gx, gy = state['goal_center']
        sx, sy = state['keeper_pos']
        dist = state['dist_to_ball']

        # Inside penalty area
        if state['is_in_box']:
            if dist < 20:
                # Intercept: move towards ball
                return {'type': 'move', 'target': state['ball_pos']}
            # Hold line around penalty spot
            return {'type': 'move', 'target': (gx + (1 if self.team == 'A' else -1) * 2, gy)}

        # Outside box
        # Stay at goal mouth
        return {'type': 'move', 'target': (gx, gy)}