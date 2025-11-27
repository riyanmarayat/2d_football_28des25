from agents.heuristic.base import BaseHeuristicAgent


class GoalkeeperAgent(BaseHeuristicAgent):
    """
    Kiper sederhana: jaga garis gawang, maju sedikit jika bola masuk kotak.
    """

    def __init__(self, team: str):
        super().__init__(team, "Goalkeeper", max_speed=5.0)

    def compute_target(self, player, ball, field):
        goal_x = 0.0 if self.team == "A" else field.width
        goal_y = field.height / 2
        box_depth = field.width * 0.15
        in_box = (goal_x <= ball.x <= goal_x + box_depth) if self.team == "A" else (goal_x - box_depth <= ball.x <= goal_x)
        # jika bola masuk kotak, maju mendekati bola; jika tidak, tetap di garis
        if in_box:
            t_x = (goal_x * 0.8 + ball.x * 0.2) if self.team == "A" else (goal_x * 0.8 + ball.x * 0.2)
            t_y = (goal_y * 0.7 + ball.y * 0.3)
        else:
            t_x = goal_x
            t_y = goal_y
        return t_x, t_y
