import random
from agents.base_agent import FootballAgent
from reinforcement_learning.dqn_agent import DQNCoreAgent

class StrikerAgent(FootballAgent):
    def __init__(self, team):
        super().__init__(team, "Striker")
        self.control_range = 0.5  # meters
        self.tackle_range = 0.5  # meters

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
        # no one controls the ball -> try to control
        #control the ball first
        self.ball_controller = ball_controller
        if state['ball_controller'] is None:
            #Action: Control the ball
            if state['dist_to_ball'] < self.control_range:
                return {'type': 'control', 'target': state['ball_pos']}
            #ActionL Move to the ball
            return {'type': 'move', 'target': state['ball_pos']}

        # I control the ball -> i can choose run, shoot or pass
        if state['ball_controller'] == 'me':
            # if close to goal and no defender in 8 units -> shoot
            if state['dist_to_goal'] < 25 and state['nearest_opponents_dist'] > 8:
                return{'type': 'shoot', 'target': state['goal_pos']}
            # if close to goal and defender in 8 units -> pass
            if state['nearest_opponents_dist'] < 5:
                return {'type': 'pass', 'target': state['best_teammate_pos']}
            #otherwise, run with the ball
            return {
                'type': 'run_with_ball',
                'target':
                    (
                        state['self_pos'][0] + (10 if self.team.upper() == "A" else -10),
                        state['self_pos'][1] + (random.randint(-5, 5)),
                    ),
            }

        # opponent controls the ball -> try to distrupt
        if state['ball_controller'] == 'opponent':
            # if wihtin tackle range -> tackle
            if state['dist_to_ball'] <= self.tackle_range:
                return {'type': 'tackle', 'target': state['ball_pos']}
            #otherwise, move to block/intecept by shadowing the ball
            return {'type': 'block', 'target': state['ball_pos']}
        if state['ball_controller'] == 'teammate':
            # if teammate has ball and close to goal -> off-ball run
            if state['dist_to_goal'] < 25:
                return {
                    'type': 'move',
                    'target':
                        (
                            state['ball_pos'][0] + (random.randint(-10, 10)),
                            state['ball_pos'][1] + (random.randint(-10, 10)),
                        ),
                }
            #otherwise, move
            return {
                'type': 'move',
                'target':
                    (
                        state['ball_pos'][0] + (random.randint(-20, 20)),
                        state['ball_pos'][1] + (random.randint(-30, 30)),
                    ),
            }
        # default action
        return {'type': 'move', 'target': state['self_pos']}