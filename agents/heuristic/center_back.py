from agents.heuristic.base import BaseHeuristicAgent


class CenterBackAgent(BaseHeuristicAgent):
    """
    Menjaga garis pertahanan: posisikan diri di antara bola dan gawang sendiri.
    """

    def __init__(self, team: str):
        super().__init__(team, "Center Back", max_speed=5.2)

    def compute_target(self, player, ball, field):
        goal_x = 0.0 if self.team == "A" else field.width
        goal_y = field.height / 2
        t_x = (ball.x + goal_x) / 2
        t_y = (ball.y + goal_y) / 2
        # jangan naik terlalu tinggi
        if self.team == "A":
            t_x = min(t_x, field.width * 0.45)
        else:
            t_x = max(t_x, field.width * 0.55)
        return t_x, t_y
