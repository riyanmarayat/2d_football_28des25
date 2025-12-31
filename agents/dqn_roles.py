from typing import Optional
from agents.striker.dqn_striker import DQNStriker


class DQNTacticsAgent(DQNStriker):
    """
    Turunan DQNStriker untuk role lain. Perbedaan utama pada default speed/sprint
    agar peran defensif lebih lambat/agresif lebih cepat. Logic belajar tetap DQN.
    """
    def __init__(
        self,
        team: str,
        player_index: int,
        side: str = "left",
        seed: Optional[int] = 0,
        max_move_speed: float = 6.0,
        sprint_multiplier: float = 1.2,
        role_name: Optional[str] = None,
    ):
        super().__init__(
            team=team,
            seed=seed,
            player_index=player_index,
            side=side,
            max_move_speed=max_move_speed,
            sprint_multiplier=sprint_multiplier,
            role_name=role_name,
        )


class DQNGoalkeeperAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=5.0, sprint_multiplier=1.05, role_name=role_name)


class DQNCenterBackAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=5.4, sprint_multiplier=1.1, role_name=role_name)


class DQNRightFullbackAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=5.8, sprint_multiplier=1.2, role_name=role_name)


class DQNLeftFullbackAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=5.8, sprint_multiplier=1.2, role_name=role_name)


class DQNCentralMidfielderAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=6.0, sprint_multiplier=1.25, role_name=role_name)


class DQNRightMidfielderAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=6.1, sprint_multiplier=1.25, role_name=role_name)


class DQNLeftMidfielderAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=6.1, sprint_multiplier=1.25, role_name=role_name)


class DQNRightWingerAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=6.4, sprint_multiplier=1.3, role_name=role_name)


class DQNLeftWingerAgent(DQNTacticsAgent):
    def __init__(self, team: str, player_index: int, side: str = "left", seed: Optional[int] = 0, role_name: Optional[str] = None):
        super().__init__(team, player_index, side=side, seed=seed, max_move_speed=6.4, sprint_multiplier=1.3, role_name=role_name)
