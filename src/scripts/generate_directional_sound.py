## import tukey for last scipy version
import time
from utils.sound_functions import compute_directional_sound
import asyncio



def play_directional_loop(user_pos, user_heading_deg, target_pos,
                          max_duration_s=20.0):
    
    # sd.default.device = 2
    t0 = time.time()
    try:
        while time.time() - t0 < max_duration_s:
            print(target_pos)
            target_pos = (2, target_pos[1] + 1.0 * 0.2)  # move target 0.1 m forward each beep
            asyncio.run(compute_directional_sound(user_pos, user_heading_deg, target_pos))

            # (In a real device update user_pos/heading here; example keeps fixed)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    # Example usage:
    user_pos = (0.0, 0.0)
    user_heading_deg = 0.0   # 0 = facing +x axis (east). Heading positive clockwise (like compass).
    # place target (2m right-front)
    target_pos = (2.0, -2.0)
    play_directional_loop(user_pos, user_heading_deg, target_pos, max_duration_s=20.0)
