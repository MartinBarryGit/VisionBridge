# Decompiled with PyLingual (https://pylingual.io)
# Internal filename: /Users/barry/Desktop/HES-SO/VisionBridge/src/utils/sound_functions.py
# Bytecode version: 3.12.0rc2 (3531)
# Source timestamp: 2025-10-01 09:44:31 UTC (1759311871)

import numpy as np
import math
import sounddevice as sd
import time
SR = 48000

async def compute_directional_sound(user_pos, user_heading_deg, target_pos):
    dist = math.hypot(target_pos[0] - user_pos[0], target_pos[1] - user_pos[1])
    rel_angle = angle_to_relative(user_pos, user_heading_deg, target_pos)
    rate = distance_to_rate(dist)
    interval = 1.0 / rate
    print(f'Distance {dist:.1f} m → {rate:.2f} beeps/s, relative angle {rel_angle:.1f}°')
    if abs(rel_angle) < 1.0:
        print('Target is straight ahead!')
        interval *= 0.5
    stereo = make_stereo_beep(rel_angle, freq=880.0, duration=0.12, amp=0.5)
    sd.play(stereo, SR)
    sd.wait()
    remaining = interval - 0.12
    if remaining > 0:
        time.sleep(remaining)

def angle_to_relative(user_pos, user_heading_deg, target_pos):
    dx = target_pos[0] - user_pos[0]
    dy = target_pos[1] - user_pos[1]
    bearing = math.degrees(math.atan2(dy, dx))
    print(f'bearing: {bearing}')
    rel = (bearing - user_heading_deg + 180) % 360 - 180
    return rel

def angle_to_pan(rel_angle_deg):
    pan = 0.5 * (1 + rel_angle_deg / 180.0)
    pan = max(0.0, min(1.0, pan))
    return pan

def equal_power_gains(pan):
    theta = pan * math.pi / 2
    left = math.cos(theta)
    right = math.sin(theta)
    beta = 50
    x = np.array([left, right]) * beta
    x = np.exp(x) / sum(np.exp(x))
    print(x)
    return (x[0], x[1])

def apply_itd_stereo(left_buf, right_buf, rel_angle_deg, sr=SR, max_itd_ms=0.7):
    max_delay_s = max_itd_ms / 1000.0
    delay = max_delay_s * math.sin(math.radians(rel_angle_deg))
    delay_samples = int(round(abs(delay) * sr))
    if delay_samples == 0:
        return (left_buf, right_buf)
    if delay > 0:
        left = np.concatenate((np.zeros(delay_samples), left_buf))[:len(left_buf)]
        right = right_buf
        return (left, right)
    right = np.concatenate((np.zeros(delay_samples), right_buf))[:len(right_buf)]
    left = left_buf
    return (left, right)

def make_tone(freq=880.0, duration=0.12, sr=44100, amp=0.6):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    env = np.hanning(len(t))
    return amp * tone * env

def make_stereo_beep(rel_angle_deg, freq=880.0, duration=0.12, amp=0.6, sr=SR):
    mono = make_tone(freq=freq, duration=duration, sr=sr, amp=amp)
    pan = angle_to_pan(rel_angle_deg)
    l_gain, r_gain = equal_power_gains(pan)
    left = mono * l_gain
    right = mono * r_gain
    left, right = apply_itd_stereo(left, right, rel_angle_deg, sr=sr)
    return np.vstack((left, right)).T

def distance_to_rate(dist_m, min_rate=0.5, max_rate=4.0, max_dist=20.0):
    d = max(0.0, min(max_dist, dist_m))
    rate = max_rate - d / max_dist * (max_rate - min_rate)
    return rate