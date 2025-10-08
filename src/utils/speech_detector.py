import sounddevice as sd
from scipy.io.wavfile import write
from collections import deque
import torch
import numpy as np

# CRITICAL: Use 16kHz for Silero VAD
sample_rate = 16000         # 16 kHz - Silero VAD requirement
chunk_size = 512            # number of samples for each VAD call (~32ms)
frame_ms = (chunk_size / sample_rate) * 1000.0  # ~32 ms

# Pre-roll amount in ms (how much audio we include before "start" is triggered)
# For example, 100 ms → about 3 frames of 32 ms each
pre_roll_ms = 100
num_pre_roll_frames = int(pre_roll_ms // frame_ms)  # ~3 frames

# Initialize VAD
print("Loading Silero VAD model...")
model_vad, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
print("VAD model loaded successfully")

    
def record_audio(save_path: str = "recorded_audio.wav"):
    
    frames = []
    recording = False
    proper_start_sent = False
    print("Listening for speech...")
    print(f"Sample rate: {sample_rate} Hz, Chunk size: {chunk_size} samples (~{frame_ms:.1f} ms)")
    vad_iterator = VADIterator(model_vad)

    # Buffers for speech segment detection
    audio_buffer = bytearray()  # accumulate raw PCM for an utterance
    triggered = False

    # This ring buffer will store recent *silent* (non-triggered) chunks.
    # We’ll prepend these once we detect a "start" event.
    ring_buffer = deque(maxlen=num_pre_roll_frames)

    # We accumulate smaller incoming frames here until we reach `chunk_size` samples
    vad_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            # CRITICAL: Record at 16kHz mono for Silero VAD
            data = sd.rec(int(5 * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
            sd.wait()  # Wait until recording is finished
            
            if not data.any():
                break

            # data is already a numpy array of shape (samples, 1), extract mono channel
            pcm_samples = data[:, 0]
            
            # Convert to float32 in range [-1, 1]
            audio_float32 = pcm_samples.astype(np.float32) / 32768.0
            vad_buffer = np.concatenate((vad_buffer, audio_float32))

            while len(vad_buffer) >= chunk_size:
                current_chunk = vad_buffer[:chunk_size]
                vad_buffer = vad_buffer[chunk_size:]
                
                # Call VAD iterator with correct sample rate
                speech_segments = vad_iterator(current_chunk, return_seconds=False)
                
                is_speech_start = (speech_segments is not None and 'start' in speech_segments)
                is_speech_end   = (speech_segments is not None and 'end'   in speech_segments)

                chunk_int16 = (current_chunk * 32768.0).astype(np.int16).tobytes()

                if is_speech_start and not triggered:
                    print(f"🎤 Speech START detected at frame {speech_segments['start']}")
                    # Prepend pre-roll frames
                    for rb_chunk in ring_buffer:
                        audio_buffer.extend(rb_chunk)
                    ring_buffer.clear()
                    triggered = True
                    proper_start_sent = False  # Reset for new utterance

                if triggered:
                    audio_buffer.extend(chunk_int16)
                    # Check if buffer meets threshold and proper start not sent
                    if len(audio_buffer) >= 24000 and not proper_start_sent:
                        print("✅ Proper speech start detected (>=0.75s)")
                        message_to_send = {'detection': 'proper_speech_start'}
                        # await websocket.send_text(str(message_to_send))
                        proper_start_sent = True
                else:
                    # Store in ring buffer for pre-roll
                    ring_buffer.append(chunk_int16)

                if is_speech_end and triggered:
                    print(f"🛑 Speech END detected at frame {speech_segments['end']}")
                    triggered = False

                    if len(audio_buffer) >= 24000:
                        # Save the recorded audio
                        audio_array = np.frombuffer(audio_buffer, dtype=np.int16)
                        write('recorded_audio.wav', sample_rate, audio_array)
                        print(f"💾 Audio saved: {len(audio_array)/sample_rate:.2f}s")
                        
                        # Clear buffer for next utterance
                        audio_buffer = bytearray()
                        proper_start_sent = False
                    return 0
    except KeyboardInterrupt:
        print("\n⏹️  Recording stopped by user")
        return frames

             
if __name__ == "__main__":
    audio_frames = record_audio()
