from utils.speech_detector import record_audio
from utils.sound_functions import SoundPlayer
from ultralytics import YOLO
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import os 
from config import parent_dir
from utils.AI_door_descriptor import AI_descriptor
from utils.door_detection_functions import box_sound_detection
import asyncio
import cv2
from unidecode import unidecode

async def main():
    player = SoundPlayer()
    model_path = os.path.join(parent_dir, "runs/detect/multi_dataset/weights/best.pt")

    model = YOLO(model_path)
    speaker = AsyncOpenAI()
    cap = cv2.VideoCapture(0)
    descriptor = AI_descriptor()
    while True:
        record_audio()
        client = OpenAI() 
        
        with open("recorded_audio.wav", "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="gpt-4o-transcribe", 
                file=f
            )
        sentence_detector = "ok antoine"
        print(unidecode(transcription.text.lower()))
        if sentence_detector in unidecode(transcription.text.lower()):
            ret, frame = cap.read()
            if not ret:
                print("Impossible de lire une frame")
                break
            user_input = transcription.text.lower().replace(sentence_detector, "").strip()
            description_response, answer = descriptor.describe_frame(user_input, frame)
            async with speaker.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=answer,
                instructions="Speak in a cheerful and positive tone.",
                response_format="pcm",
            ) as response:
                    await LocalAudioPlayer().play(response)
            if description_response == 0:
                print("Response received successfully")
                while True:
                    ret, frame = cap.read()
                    box_sound_detection(frame, model, player)
                    
            print("Command recognized: OK Etienne")
        


if __name__ == "__main__":
    asyncio.run(main())