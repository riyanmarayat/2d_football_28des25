import math

def goalkeeper_rules(agent, state):
    """
    Rule-based decision system for a goalkeeper.
    Focuses on blocking shots, staying near goal, and clearing the ball.
    """
    ax, ay = agent["x"], agent["y"]
    bx, by = state["ball_position"]
    distance_to_ball = math.hypot(ax - bx, ay - by)

    if distance_to_ball < 1.0:
        return "clear"
    elif state.get("ball_heading_to_goal", False):
        return "dive"
    elif abs(ax - state.get("goal_x", 0)) > 1.5:
        return "move_to_goal"
    else:
        return "hold_position"