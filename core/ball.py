import math

FRICTION = {
    "dry-grass-standard" : 0.99,
    "wet-grass" : 0.995,
    "very-wet-grass" : 0.96,
    "rough-sandy" : 0.92,
    "synthetic" : 0.98,
    "ice-superslippery" : 0.999,
}
FIELD = (100, 75) # Field size in meters (width, height)
class Ball:
    def __init__(self, field_width=FIELD[0], field_height=FIELD[1], condition="dry"):
        # Inital position = center of the field
        self.x, self.y = field_width/2, field_height/2
        self.vx, self.vy = 0, 0 # Velocity (m/s)

        # Ball physic properties
        self.radius = 0.11 # Standar bola yang digunakan dalam pertandingan profesional
        self.friction = self._get_friction(condition) # Friction coefficient dry-grass-standard=0.99

        # Field boundaries
        self.field_width = field_width
        self.field_height = field_height

        self.dt = 1.0

    def _get_friction(self, condition):
        return {
            "dry": 0.985,
            "wet": 0.995,
            "heavy_wet": 0.96,
            "sand": 0.92,
            "synthetic": 0.98,
            "ice": 0.999,
        }.get(condition, 0.985) # Default to dry friction if condition not found

    def update(self, dt=None, field=None): #dt is time delta default 1.0 but can be 0.1 for 10FPS etc (10 FPS berarti 10 gambar/frame per detik)
        """
        Update ball position, check for goals, then apply friction and boundary-clamping.
        :param dt:
        :param field:
        :return:
        """
        if dt is not None:
            self.dt = dt
        #move the ball according current velocity
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        #check for tgoal before claamping back inside the pitch
        if field is not None:
            goal = field.is_goal(self.x, self.y)
            if goal:
                #announce and zero velocity so ball stops in net
                print(f"Goal for the {goal} team!")
                self.vx = 0.0
                self.vy = 0.0
                return #skip clamping so ball can reset in the goal

        #clamp within  the playing area
        # # ball.radius away from sidelines and endlines
        # self.x =  max(self.radius, min(self.x, self.field_width - self.radius))
        # self.y =  max(self.radius, min(self.y, self.field_height - self.radius))

        self.vx *= self.friction
        self.vy *= self.friction

    def kick(self, power, angle_deg):
        """Kick the ball in a direction with a certain power and angle"""
        # convert power from km/h to m/s
        power = power * 1000 / 3600
        rad = math.radians(angle_deg)
        self.vx = round(power * math.cos(rad), 2)
        self.vy = round(power * math.sin(rad), 2)

    def kick_towards(self, tx, ty, power):
        dx = tx - self.x
        dy = ty - self.y
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < 1e-6:
            return
        nx = dx / dist
        ny = dy / dist
        self.vx = nx * power
        self.vy = ny * power

    def get_position(self):
        """Get the current position of the ball"""
        return (round(self.x, 2), round(self.y, 2))

    def reset(self):
        self.x = self.field_width / 2
        self.y = self.field_height / 2
        self.vx = 0.0
        self.vy = 0.0