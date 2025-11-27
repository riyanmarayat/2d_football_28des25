import math
from typing import Tuple, Dict, Any


def _clamp_speed(dx: float, dy: float, max_speed: float) -> Tuple[float, float]:
    d = math.sqrt(dx * dx + dy * dy)
    if d < 1e-6:
        return 0.0, 0.0
    scale = min(1.0, max_speed / d)
    return dx * scale, dy * scale


class BaseHeuristicAgent:
    """
    Agen sederhana berbasis vektor kecepatan.
    Mengembalikan desired_velocity() sehingga simulator langsung memakai vx/vy.
    """

    def __init__(self, team: str, role: str, max_speed: float = 5.5):
        self.team = team.upper()
        self.role = role
        self.max_speed = max_speed
        self._target: Tuple[float, float] = (0.0, 0.0)

    def desired_velocity(self, player: Dict[str, Any], ball, field) -> Tuple[float, float]:
        tx, ty = self.compute_target(player, ball, field)
        self._target = (tx, ty)
        dx = tx - player["x"]
        dy = ty - player["y"]
        return _clamp_speed(dx, dy, self.max_speed)

    # To be implemented by subclasses
    def compute_target(self, player: Dict[str, Any], ball, field) -> Tuple[float, float]:
        raise NotImplementedError


class CenterBackAgent(BaseHeuristicAgent):
    """
    Menjaga garis pertahanan, bergerak ke antara bola dan gawang sendiri.
    """

    def __init__(self, team: str):
        super().__init__(team, "Center Back", max_speed=5.2)

    def compute_target(self, player, ball, field):
        goal_x = 0.0 if self.team == "A" else field.width
        goal_y = field.height / 2
        # titik di antara bola dan gawang, tapi tidak melewati gawang
        t_x = (ball.x + goal_x) / 2
        t_y = (ball.y + goal_y) / 2
        # jaga agar tidak terlalu tinggi: max 40% lapangan dari gawang sendiri
        if self.team == "A":
            t_x = min(t_x, field.width * 0.45)
        else:
            t_x = max(t_x, field.width * 0.55)
        return t_x, t_y


class CentralMidfielderAgent(BaseHeuristicAgent):
    """
    Menjaga tengah, mendukung serangan/bertahan tergantung posisi bola.
    """

    def __init__(self, team: str):
        super().__init__(team, "Central Midfielder", max_speed=5.8)

    def compute_target(self, player, ball, field):
        mid_y = field.height * 0.5
        # dorong ke depan jika bola di depan, mundur jika bola di belakang
        if self.team == "A":
            t_x = max(field.width * 0.35, min(ball.x - 3.0, field.width * 0.65))
        else:
            t_x = min(field.width * 0.65, max(ball.x + 3.0, field.width * 0.35))
        t_y = (ball.y * 0.6 + mid_y * 0.4)
        return t_x, t_y


class RightFullbackAgent(BaseHeuristicAgent):
    """
    Bek kanan: jaga sayap kanan, tutup jalur bola, overlap sedikit.
    """

    def __init__(self, team: str):
        super().__init__(team, "Right Fullback", max_speed=5.6)

    def compute_target(self, player, ball, field):
        flank_y = field.height * 0.75
        if self.team == "A":
            base_x = field.width * 0.25
            t_x = min(base_x + (ball.x * 0.3), field.width * 0.55)
        else:
            base_x = field.width * 0.75
            t_x = max(base_x - (field.width - ball.x) * 0.3, field.width * 0.45)
        t_y = (ball.y * 0.5 + flank_y * 0.5)
        return t_x, t_y


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


class RightWingerAgent(BaseHeuristicAgent):
    """
    Winger kanan: tetap lebar, maju seiring bola, potong ke tengah di sepertiga akhir.
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


class LeftWingerAgent(BaseHeuristicAgent):
    """
    Winger kiri: tetap lebar, maju seiring bola, potong ke tengah di sepertiga akhir.
    """

    def __init__(self, team: str):
        super().__init__(team, "Left Winger", max_speed=6.0)

    def compute_target(self, player, ball, field):
        wide_y = field.height * 0.2
        if self.team == "A":
            t_x = max(ball.x + 5.0, field.width * 0.55)
        else:
            t_x = min(ball.x - 5.0, field.width * 0.45)
        t_y = (ball.y * 0.4 + wide_y * 0.6)
        return t_x, t_y
