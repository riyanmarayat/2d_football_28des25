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
    Agen sederhana berbasis target posisi.
    Simulator akan memanggil desired_velocity(), jadi kita set vx/vy langsung.
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

    # Harus diimplementasikan subclass
    def compute_target(self, player: Dict[str, Any], ball, field) -> Tuple[float, float]:
        raise NotImplementedError
