import pygame
import numpy as np
import os
import glob
import sys
import re

# --- CONFIGURATION ---
REPLAY_FOLDER = "replays/muzero-tron-run-v10"
PLAYBACK_FPS = 20 # Speed of the Tron snake (Frames per second)
UI_FPS = 60  # Speed of the window (Keeps keyboard inputs instant!)
CELL_SIZE = 30  # Visual size of the grid squares

# Matched exactly to your TronEnv
COLORS = {
    "background": (34, 45, 61),
    "head": (252, 186, 3),
    "body": (0, 247, 255),
    "wall": (255, 255, 255),
    "grid_line": (50, 50, 50),
}


def draw_state(screen, state):
    """Draws a single (grid_h, grid_w, 3) observation frame."""
    screen.fill(COLORS["background"])

    grid_h, grid_w, _ = state.shape

    for y in range(grid_h):
        for x in range(grid_w):
            rect = pygame.Rect(
                x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE - 1, CELL_SIZE - 1
            )

            # Channel 2: Walls (Borders and random walls)
            if state[y, x, 2] == 1.0:
                pygame.draw.rect(screen, COLORS["wall"], rect)

            # Channel 1: Trail/Body
            elif state[y, x, 1] == 1.0:
                if state[y, x, 0] == 1.0:
                    pygame.draw.rect(screen, COLORS["head"], rect)
                else:
                    pygame.draw.rect(screen, COLORS["body"], rect)

            # Channel 0: Head (Fallback)
            elif state[y, x, 0] == 1.0:
                pygame.draw.rect(screen, COLORS["head"], rect)

    # Draw grid lines
    for x in range(0, grid_w * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(
            screen, COLORS["grid_line"], (x, 0), (x, grid_h * CELL_SIZE), 1
        )
    for y in range(0, grid_h * CELL_SIZE, CELL_SIZE):
        pygame.draw.line(
            screen, COLORS["grid_line"], (0, y), (grid_w * CELL_SIZE, y), 1
        )

    pygame.display.flip()


def play_replay(file_path, screen, clock):
    print(f"\n▶️ Ready: {os.path.basename(file_path)} (Press SPACE to start)")
    try:
        states = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return True

    frame_idx = 0
    num_frames = len(states)
    paused = True
    pygame.event.clear()

    time_accumulator = 0.0
    time_per_frame = 1000.0 / PLAYBACK_FPS

    # --- THE COMPASS HACK ---
    # Grab the walls (Channel 2) from Frame 0. This is our absolute "True North" reference.
    frame0_walls = states[0][:, :, 2]

    while frame_idx < num_frames:
        dt = clock.tick(UI_FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_n:
                    return True
                elif event.key == pygame.K_SPACE:
                    paused = not paused

            if event.type == pygame.KEYDOWN and paused:
                if event.key == pygame.K_RIGHT:
                    frame_idx = min(frame_idx + 1, num_frames - 1)
                elif event.key == pygame.K_LEFT:
                    frame_idx = max(frame_idx - 1, 0)

        # --- UN-ROTATE THE FRAME ---
        current_frame = states[frame_idx]
        current_walls = current_frame[:, :, 2]

        # Test all 4 possible rotations to find which one aligns the walls with Frame 0
        best_k = 0
        for k in range(4):
            if np.array_equal(np.rot90(current_walls, k=k, axes=(0, 1)), frame0_walls):
                best_k = k
                break

        # Apply the correcting rotation to the entire 3-channel frame
        static_frame = np.rot90(current_frame, k=best_k, axes=(0, 1))

        # Draw the un-rotated frame!
        draw_state(screen, static_frame)

        if not paused:
            time_accumulator += dt
            if time_accumulator >= time_per_frame:
                frame_idx += 1
                time_accumulator = 0.0

                if frame_idx >= num_frames:
                    print("💀 Game Over. Press 'N' for next, or 'SPACE' to review.")
                    paused = True
                    frame_idx = num_frames - 1

    return True


def main():
    pygame.init()

    replay_files = glob.glob(os.path.join(REPLAY_FOLDER, "*.npy"))

    def sort_key(f):
        match = re.search(r"gen_(\d+)", f)
        return int(match.group(1)) if match else 0

    replay_files.sort(key=sort_key)

    if not replay_files:
        print(f"❌ No .npy files found in '{REPLAY_FOLDER}' folder.")
        sys.exit()

    sample_states = np.load(replay_files[0], allow_pickle=True)
    grid_h, grid_w, _ = sample_states[0].shape
    screen = pygame.display.set_mode((grid_w * CELL_SIZE, grid_h * CELL_SIZE))
    pygame.display.set_caption("Tron AI Brain Viewer")
    clock = pygame.time.Clock()

    print(f"Found {len(replay_files)} replays.")
    print("\n🎮 CONTROLS:")
    print(" - SPACE: Pause / Play")
    print(" - LEFT/RIGHT ARROWS: Step backward/forward (while paused)")
    print(" - N: Skip to Next replay")
    print(" - Q / ESC: Quit")

    for file_path in replay_files:
        keep_going = play_replay(file_path, screen, clock)
        if not keep_going:
            break

    pygame.quit()


if __name__ == "__main__":
    main()
