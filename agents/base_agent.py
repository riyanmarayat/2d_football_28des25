class FootballAgent:
    def __init__(self, team, role):
        self.team = team # Team A or B
        self.role = role # Role: goalkeeper, defender, midfielder, striker
        self.ball_controller = None # Player who controls the ball
        self.action_delay = 4 # 4 Delay in step for 15 FPS like 1 step is 0.0667 second and average player reaction 0.18-0.25 second

    def get_state(self, self_player, players, ball, field):
        """
        Return the relevant state for decision making.
        env: simulation environment (field, ball, all players, etc.)
        self_player: reference to this agent player objec
        """

        raise NotImplementedError

    def decide_action(self, state, ball_controller=None):
        """
        Decide what action to take in current state.
        (Rule-based for now, but can upgrade to ML.)
        """
        if self.ball_controller is None:
            self.ball_controller = ball_controller
        if state['dist_to_ball'] < 1:
            if self.ball_controller is not None:
                return {'type': 'control'}

        raise NotImplementedError
