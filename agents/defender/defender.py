import math

def defender_rules(agent, state):
    """
    Rule-based decision for a defender.
    Focuses on intercepting, clearing the ball, and marking opponents.
    """
    ax, ay = agent["x"], agent["y"]
    bx, by = state["ball_position"]
    distance_to_ball = math.hypot(ax - bx, ay - by)

    if distance_to_ball < 1.5:
        return "clear"
    elif state.get("opponent_near_goal", False):
        return "mark"
    else:
        return "return_to_defense"