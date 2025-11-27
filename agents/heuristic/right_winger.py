from agents.heuristic.base import BaseHeuristicAgent


class RightWingerAgent(BaseHeuristicAgent):
    """
    Winger kanan: tetap lebar, maju seiring bola, siap cut inside.
    """

    def __init__(self, team: str):
        super().__init__(team, "Right Winger", max_speed=6.0)

    def compute_target(self, player, ball, field):
        wide_y = field.height * 0.8
        if self.team == "A":
            t_x = max(ball.x + 5.0, field.width * 0.55)
        else:
            t_x = min(ball.x - 5.0, field.width * 0.45)
        t_y = (ball.y * 0.4 + wide_y * 0.6)
        return t_x, t_y
