FIELD = (100, 75) # Field size in meters (width, height)

class Field:
    def __init__(self, width=FIELD[0], height=FIELD[1]):
        """
                Initializes a football field.
                :param width: Width of the field in meters.
                :param height: Height of the field in meters.
                """
        self.width = width
        self.height = height

        # Extra boundary zone (for thwrow-ins, corners, etc.)
        self.outer_margin = 3.0 # meters

        # Total filed dimensions including margin
        self.total_width = self.width + 2 * self.outer_margin
        self.total_height = self.height + 2 * self.outer_margin

        # Define goal area dimensions (standard ~7.32m wide)
        goal_width = 7.32
        goal_height = 2.44

        # Left and right goal posts
        self.left_goal = {
            "x": 0.0,
            "y_top": (self.height - goal_width) / 2,
            "y_bottom": (self.height + goal_width) / 2,
        }

        self.right_goal = {
            "x": self.width,
            "y_top": (self.height - goal_width) / 2,
            "y_bottom": (self.height + goal_width) / 2,
        }

    def is_goal(self, x, y):
        """
        Check if the ball position (x, y) is a goal.
        :return: 'left', 'right', or None
        """
        if x <= 0.0:
            if self.left_goal["y_top"] <= y <= self.left_goal["y_bottom"]:
                return "left"
        elif x >= self.width:
            if self.right_goal["y_top"] <= y <= self.right_goal["y_bottom"]:
                return "right"
        return None

    def in_bounds(self, x, y, radius=0.0):
        """
        Check if a position is within the field, considering optional radius.
        :return: True if inside, False otherwise
        """
        return (
                radius <= x <= self.width - radius and
                radius <= y <= self.height - radius
        )

    def in_outer_zone(self, x, y):
        """
        Check if a position is within the outer zone of the field.
        :return: True if inside, False otherwise
        """
        return (
                -self.outer_margin <= x <= self.width + self.outer_margin and
                -self.outer_margin <= y <= self.height + self.outer_margin
                and not self.in_bounds(x, y)
        )

    def get_goal_pos(self, team, own=False):
        """
        Get the goal position for a given team.
        :param team: 'A' or 'B'
        :param own: True if own goal, False if opponent's goal
        :return: (x, y) coordinates of the goal
        """
        if team == "A":
            return (self.left_goal["x"], self.left_goal["y_top"]) if own else (self.right_goal["x"], self.right_goal["y_top"])
        elif team == "B":
            return (self.right_goal["x"], self.right_goal["y_top"]) if own else (self.left_goal["x"], self.left_goal["y_top"])
        else:
            raise ValueError("Invalid team. Use 'A' or 'B'.")