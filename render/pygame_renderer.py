import pygame
import math

class PygameRenderer:
    def __init__(self, width=100, height=75, scale=10, outer_margin=3.0, show_vertical_zones=False, show_horizontal_zones=False):
        pygame.init()
        self.scale = scale
        self.width = width
        self.height = height
        self.outer_margin = outer_margin
        self.surface_width = int((width + 2 * outer_margin) * scale)
        self.surface_height = int((height + 2 * outer_margin) * scale)
        self.screen = pygame.Surface((self.surface_width, self.surface_height))
        self.show_vertical_zones = show_vertical_zones
        self.show_horizontal_zones = show_horizontal_zones
        self.font = pygame.font.SysFont("Arial", 12)

    def draw_arc(self, center, radius, start_angle_deg, sweep_deg, color, width=2):
        points = []
        step = 2

        for angle in range(start_angle_deg, start_angle_deg + sweep_deg + 1, step):
            rad = math.radians(angle % 360)
            x = int(center[0] + radius * math.cos(rad))
            y = int(center[1] + radius * math.sin(rad))
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, width)

    def get_horizontal_zones(self, y_world):
        """Return which horizontal zone the y coordinate belongs to."""
        field_zones = [
            (0.0, 0.23, "Wing Left"),
            (0.23, 0.4, "Half-Space Left"),
            (0.4, 0.6, "Center Channel"),
            (0.6, 0.77, "Half-Space Right"),
            (0.77, 1.0, "Wing Right")
        ]
        y_ratio = y_world / self.height
        for low, high, name in field_zones:
            if low <= y_ratio < high:
                return name
        return "Unknown"

    def get_vertical_third(self, x_world, team='left'):
        ratio = x_world / self.width
        if team == 'left':
            if ratio < 1/3:
                return "Defensive Third"
            elif ratio < 2/3:
                return "Middle Third"
            else:
                return "Attacking Third"
        elif team == 'right':
            if ratio < 1/3:
                return "Attacking Third"
            elif ratio < 2/3:
                return "Middle Third"
            else:
                return "Defensive Third"
        return "Unknown"

    def log_player_zones(self, state, current_time, save_log=False):
        for player in state["players"]:
            team = player["team"]
            x, y = player["x"], player["y"]
            h_zone = self.get_horizontal_zones(y)
            v_zone = self.get_vertical_third(x, team='left' if team == 'A' else 'right')
            if save_log==False:
                print(f"Time {current_time:.2f}s | Team {team} | Pos ({x:.2f}, {y:.2f}) -> {h_zone}, {v_zone}")
            else:
                with open("zones_log.txt", "a") as f:
                    f.write(f"Time {current_time:.2f}s | Team {team} | Pos ({x:.2f}, {y:.2f}) -> {h_zone}, {v_zone}\n")


    def render(self, state):
        self.screen.fill((0, 102, 0))  # outer zone (dark green)

        # draw main field (lighter green)
        field_color = (34, 139, 34)
        field_rect = pygame.Rect(
            self.outer_margin * self.scale,
            self.outer_margin * self.scale,
            self.width * self.scale,
            self.height * self.scale
        )
        pygame.draw.rect(self.screen, field_color, field_rect)

        #draw white field lines
        white = (255, 255, 255)
        red = (255, 0, 0)
        om = self.outer_margin * self.scale
        w = self.width * self.scale
        h = self.height * self.scale
        center_x = om + w // 2
        center_y = om + h // 2

        #outer boundary
        pygame.draw.rect(self.screen, white, field_rect, 2)

        #center line
        pygame.draw.line(self.screen, white, (center_x, om), (center_x, om + h), 2)

        #center circle and kickoff dot
        pygame.draw.circle(self.screen, white, (center_x, center_y), int(9.15 * self.scale), 2)
        pygame.draw.circle(self.screen, white, (center_x, center_y), 2)

        #penalty boxes (approximate dimensions)
        penalty_w = 16.5 * self.scale
        penalty_h = 40.3 * self.scale
        left_penalty_rect = pygame.Rect(om, center_y - penalty_h / 2, penalty_w, penalty_h)
        right_penalty_rect = pygame.Rect(om + w - penalty_w, center_y - penalty_h / 2, penalty_w, penalty_h)
        pygame.draw.rect(self.screen, white, left_penalty_rect, 2)
        pygame.draw.rect(self.screen, white, right_penalty_rect, 2)

        #six-yard boxes (approximate dimensions)
        goal_area_w = 5.5 * self.scale
        goal_area_h = 18.32 * self.scale
        left_goal_area_rect = pygame.Rect(om, center_y - goal_area_h / 2, goal_area_w, goal_area_h)
        right_goal_area_rect = pygame.Rect(om + w - goal_area_w, center_y - goal_area_h / 2, goal_area_w, goal_area_h)
        pygame.draw.rect(self.screen, white, left_goal_area_rect, 2)
        pygame.draw.rect(self.screen, white, right_goal_area_rect, 2)

        #draw penalty spots
        left_spot = (int(om + 11 * self.scale), int(center_y))
        right_spot = (int(om + w - 11 * self.scale), int(center_y))
        pygame.draw.circle(self.screen, white, left_spot, 2)
        pygame.draw.circle(self.screen, white, right_spot, 2)

        #penalty arcs
        arc_radius = 9.15 * self.scale
        self.draw_arc(left_spot, arc_radius, 307, 107, white)
        self.draw_arc(right_spot, arc_radius, 128, 107, white)

        #corners arcs
        corner_radius = 1 * self.scale # 1 meter radius
        corners = [
            (om, om, 0),    #top-left
            (om + w, om, 90), #top-right
            (om + w, om + h, 180), #bottom-right
            (om, om + h, 270) #bottom-left
        ]
        for (cx, cy, start_angle) in corners:
            self.draw_arc((cx, cy), corner_radius, start_angle, 90, white)

        #draw tactical zones
        if self.show_vertical_zones:
            third_lines_x = [
                om + int(self.width / 3 * self.scale),
                om + int(self.width * 2 / 3 * self.scale)
            ]
            for x in third_lines_x:
                pygame.draw.line(self.screen, red, (x, om), (x, om + h), 1)

        if self.show_horizontal_zones:
            zone_fractions = [0.0, 0.23, 0.4, 0.6, 0.77, 1.0] # 5 zones between 6 boundaries
            for i in range(1, len(zone_fractions) - 1):
                y = om + int(self.height * zone_fractions[i] * self.scale)
                pygame.draw.line(self.screen, red, (om, y), (om + w, y), 1)


        # Draw goals outside field line
        goal_color = white
        pygame.draw.rect(self.screen, goal_color, pygame.Rect(
            (self.outer_margin - 1) * self.scale,
            (self.height / 2 - 3.66 + self.outer_margin) * self.scale,
            1 * self.scale,
            7.32 * self.scale
        ))
        pygame.draw.rect(self.screen, goal_color, pygame.Rect(
            (self.outer_margin + self.width) * self.scale,
            (self.height / 2 - 3.66 + self.outer_margin) * self.scale,
            1 * self.scale,
            7.32 * self.scale
        ))

        # Draw players
        for player in state["players"]:
            color = (0, 0, 255) if player["team"] == "A" else (255, 0, 0)
            x = int((player["x"] + self.outer_margin) * self.scale)
            y = int((player["y"] + self.outer_margin) * self.scale)
            pygame.draw.circle(self.screen, color, (x, y), int(0.5 * self.scale))
            label = player["role"][0].upper()
            text_surf = self.font.render(label, True, (255, 255, 255))
            text_rect = text_surf.get_rect(center=(x, y - int(0.5 * self.scale)))
            self.screen.blit(text_surf, text_rect)

        # Draw ball
        bx, by = state["ball"]
        ball_x = int((bx + self.outer_margin) * self.scale)
        ball_y = int((by + self.outer_margin) * self.scale)
        pygame.draw.circle(self.screen, (255, 255, 0), (ball_x, ball_y), int(0.3 * self.scale))

        return self.screen.copy()

    def quit(self):
        pygame.quit()
