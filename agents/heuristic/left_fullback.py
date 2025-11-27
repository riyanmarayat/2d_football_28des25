from agents.heuristic.base import BaseHeuristicAgent


class LeftFullbackAgent(BaseHeuristicAgent):
    """
    Bek kiri: jaga sayap kiri, tutup jalur bola, overlap sedikit.
    """

    def __init__(self, team: str):
        super().__init__(team, "Left Fullback", max_speed=5.6)

    def compute_target(self, player, ball, field):
        flank_y = field.height * 0.25
        if self.team == "A":
            base_x = field.width * 0.25
            t_x = min(base_x + (ball.x * 0.3), field.width * 0.55)
        else:
            base_x = field.width * 0.75
            t_x = max(base_x - (field.width - ball.x) * 0.3, field.width * 0.45)
        t_y = (ball.y * 0.5 + flank_y * 0.5)
        return t_x, t_y
