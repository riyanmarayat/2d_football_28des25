import math

def decide_action(agent, state):
    """
        Simple rule-based decision system for a football agent.
        Input:
        - agent: dict with agent info (position, role, etc.)
        - state: dict with current environment state (ball position, teammates, etc.)

        Returns:
        - str: action ('move', 'shoot', 'pass', etc.)
    """
    # Example placeholder rule: if close to ball, shoot
    ax, ay = agent["x"], agent["y"]
    bx, by = state["ball_position"]
    distance_to_ball = math.hypot(ax - bx, ay - by)

    if distance_to_ball < 1.5:
        if agent["role"] == "striker":
            return "shoot"
        elif agent["role"] == "midfielder":
            return "pass"
        elif agent["role"] == "defender":
            return "clear"
    else:
        return "move_to_ball"

def evaluate_context(state, players, agent):
    ax, ay = agent["x"], agent["y"]
    team = agent["team"]
    enemy_team = "B" if team == "A" else "A"

    def distance(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1], p2[1])

    under_pressure = any(
        player["team"] == enemy_team and distance((ax, ay), (player["x"], player["y"])) < 5
        for player in players
    )

    goal = (state["goal_x"], state["goal_y"])
    has_open_shot = True
    for player in players:
        if player["team"] == enemy_team:
            px, py = player["x"], player["y"]
            d = abs((goal[1] - ay) * px - (goal[0] - ax) * py + goal[0]*ay - goal[1]*ax) / distance((ax, ay), goal)
            if d < 3:
                has_open_shot = False
                break

    teammate_open = any(
        player["team"] == team and player != agent and all(
            distance((player["x"], player["y"]), (enemy["x"], enemy["y"])) > 5
            for enemy in players if enemy ["team"] == enemy_team
        )
        for player in players
    )

    state["under_pressure"] = under_pressure
    state["has_open_shot"] = has_open_shot
    state["teammate_open"] = teammate_open
    return state