import math

PLAYER_RADIUS = 1.2          # meter (sesuaikan skala internal)
PLAYER_MAX_PUSH = 0.6        # seberapa kuat dorongan pemisahan
RESTITUTION = 0.15           # pantulan kecil
FRICTION = 0.90              # redam setelah tumbukan

def resolve_player_collisions(players):
    n = len(players)
    for i in range(n):
        pi = players[i]
        for j in range(i+1, n):
            pj = players[j]
            dx = pj['x'] - pi['x']
            dy = pj['y'] - pi['y']
            dist_sq = dx*dx + dy*dy
            min_dist = PLAYER_RADIUS * 2.0
            if dist_sq == 0:
                # hindari nol; geser kecil
                dx, dy = 0.001, 0.0
                dist_sq = dx*dx
            if dist_sq < min_dist*min_dist:
                dist = math.sqrt(dist_sq)
                overlap = (min_dist - dist)
                nx = dx / dist
                ny = dy / dist
                # posisi digeser masing-masing
                push = overlap * 0.5
                pi['x'] -= nx * push
                pi['y'] -= ny * push
                pj['x'] += nx * push
                pj['y'] += ny * push
                # komponen relatif ke normal
                rvx = pj['vx'] - pi['vx']
                rvy = pj['vy'] - pi['vy']
                rel_norm = rvx*nx + rvy*ny
                if rel_norm < 0:
                    j_impulse = -(1 + RESTITUTION) * rel_norm
                    pi['vx'] -= j_impulse * nx * 0.5
                    pi['vy'] -= j_impulse * ny * 0.5
                    pj['vx'] += j_impulse * nx * 0.5
                    pj['vy'] += j_impulse * ny * 0.5
                # friction/damping
                pi['vx'] *= FRICTION
                pi['vy'] *= FRICTION
                pj['vx'] *= FRICTION
                pj['vy'] *= FRICTION
                # batas dorongan
                limit_speed(pi)
                limit_speed(pj)

def limit_speed(p):
    speed = (p['vx']**2 + p['vy']**2)**0.5
    max_speed = 5.5  # m/s (estimasi sprint)
    if speed > max_speed:
        scale = max_speed / speed
        p['vx'] *= scale
        p['vy'] *= scale