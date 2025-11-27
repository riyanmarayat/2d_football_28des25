for step in range(DURATION_STEPS):
    dt = 1.0 / FPS
    simulator.step(dt=dt) #update logic and physuics
    # learning
    new_state = qpa0_striker.get_state(players[0], players, ball, field)
    if old_state is None:
        old_state = new_state
    # compute reward and update Q-Table
    reward = computer_striker_reward(simulator, qpa0_striker, old_state, new_state, qpa0_striker.last_action_dict)
    print(f"Step {step+1}/{DURATION_STEPS} reward: {reward} || {qpa0_striker.last_action_dict} || {qpa0_striker.ball_controller}")
    qpa0_striker.learn(reward, new_state)
    total_reward += reward
    old_state = new_state
    if simulator.last_goal:
        break
    elif simulator.out_of_bounds:
        print(f"Ball out of bounds at ({simulator.ball.x:.2f}, {simulator.ball.y:.2f})")
        break
    elif simulator.offside:
        print("Offside! Ending this play")
        break

    frame = renderer.render(simulator.snapshot())
    exporter.add_frame(frame)
    current_time += dt
