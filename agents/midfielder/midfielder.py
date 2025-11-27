import math

def midfielder_rules(agent, state):
    """
    Rule-based decision system for a midfielder.
    Prioritizes passing, supports positioning, and ball control.
    """
    ax, ay = agent["x"], agent["y"]
    bx, by = state["ball_position"]
    distance_to_ball = math.hypot(ax - bx, ay - by)

    if distance_to_ball < 1.5:
        if state.get("teammate_open", False):
            return "pass"
        else:
            return "dribble"
    elif state.get("support_needed", False):
        return "move_to_support"
    else:
        return "move_to_ball"