import random
import time

from envs.football_aec.football_env import FootballEnv


def main():
    # Inisialisasi environment
    env = FootballEnv()
    obs, info = env.reset()

    # Tracker reward tiap agen
    total_rewards = {agent: 0.0 for agent in env.agents}

    # Loop hingga episode selesai
    for agent in env.agent_iter():
        # Dapatkan data terakhir untuk agent ini
        observation, reward, terminated, truncated, info = env.last()

        # Pilih aksi (contoh: random)
        action = env.action_spaces[agent].sample()

        # Terapkan aksi dan perbarui environment
        env.step(action)

        # Akumulasi reward
        total_rewards[agent] += reward

        # Render frame (opsional)
        env.render()
        time.sleep(1 / 15)  # sesuaikan FPS = 15

        # Hentikan jika episode berakhir
        if terminated or truncated:
            break

    # Tutup environment
    env.close()

    # Tampilkan ringkasan reward
    print("\n=== Episode Selesai ===")
    for agent, rew in total_rewards.items():
        print(f"{agent}: {rew:.2f}")


if __name__ == '__main__':
    main()
