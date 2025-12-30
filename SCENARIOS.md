# Scenario Bank (Training)

Deterministik via `seed` + episode ke-*n*:
- `seed = 0`  → selalu pakai skenario default (formasi awal saat ini).
- `seed > 0` → skenario diacak deterministik memakai `seed + episode`.
- `persentase skenario acak` (0-100) → peluang sebuah episode memakai skenario acak dibanding default. Contoh: 50 berarti ± separuh episode acak, sisanya default. Jika `seed=0`, nilai ini diabaikan (selalu default).

### Input baru di `main.py`
- `Seed skenario (0=default) [default 0]`
- `Persentase episode yang gunakan skenario acak vs default (0-100) [default 50]`

### Template skenario yang tersedia
- `kickoff_high_press` — Kickoff dengan pressing tinggi tim kanan, tim kiri lebih kompak di belakang.
- `counter_attack_left` — Tim kiri baru merebut bola di tengah, runner maju.
- `low_block_right` — Tim kanan bertahan blok rendah, kiri menguasai sepertiga akhir.
- `corner_left_attack` — Situasi sepak pojok menyerang untuk tim kiri.
- `goal_kick_right_build` — Tendangan gawang kanan, build-up pendek vs press.
- `wide_switch_left` — Bola di sayap kiri, kiri bersiap switch.
- `penalty_box_scramble` — Bola liar di kotak penalti kanan.
- `fast_break_right` — Serangan balik cepat tim kanan dari tengah.
- `midfield_press_trap` — Keduanya menyiapkan pressing jebakan di tengah.

### Bagaimana variasi bekerja
- Setiap template diberi jitter posisi kecil (spread) sehingga **1 seed × banyak episode** bisa menghasilkan puluhan ribu kombinasi unik (ball position + sebaran pemain).
- Peluang memakai skenario acak dikontrol oleh persentase input; sisanya memakai formasi default. Gunakan `seed` yang sama untuk mengulang distribusi skenario.

### Lokasi kode
- Generator: `core/scenario_bank.py` (`sample_episode_scenario`)
- Integrasi di loop episode: `main.py` (lihat pemanggilan `sample_episode_scenario` dan set `Ball` awal).
