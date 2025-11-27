from agents.heuristic.base import BaseHeuristicAgent


class CentralMidfielderAgent(BaseHeuristicAgent):
    """
    Isi tengah: naik/turun mengikuti posisi bola, jaga keseimbangan.
    """

    def __init__(self, team: str):
        super().__init__(team, "Central Midfielder", max_speed=5.8)

    def compute_target(self, player, ball, field):
        mid_y = field.height * 0.5
        if self.team == "A":
            t_x = max(field.width * 0.35, min(ball.x - 3.0, field.width * 0.65))
        else:
            t_x = min(field.width * 0.65, max(ball.x + 3.0, field.width * 0.35))
        t_y = (ball.y * 0.6 + mid_y * 0.4)
        return t_x, t_y
