from utils.llm_assistants import DoorDetectionResponse, get_agent
import numpy as np
import cv2
import json
import base64
class AI_descriptor:
    def __init__(self):
        self.agent = get_agent(format=DoorDetectionResponse)
        self.system_prompt = """You are an AI vision assistant that helps visually impaired users by analyzing images they provide.
        You will receive images encoded in base64 format. Your task is to determine if there is a door in the image.
        If you detect a door, please provide its position relative to the user (left, right, middle) and whether the door is open or closed.
        If no door is detected, respond accordingly.
        Be concise and clear in your responses.
        If there is multiple doors, describe each one briefly and ask the user which one they want to go to and ask for their preference.
        After the user selects a door do not list the other doors anymore.
        If there is no door, ask the user to maybe  turn around or take another picture.
        Once the list is composed of only one door, tell the user that you will guide them to the door."""
        self.message_history = []
        self.message_history.append({
            "role": "system",
            "content": self.system_prompt,
        })
# image_path = "/home/martin-barry/Desktop/HES-SO/VisionBridge/dataset/Doors_Merged/images/train/train_016119.jpg"
    def describe_frame(self, user_input, image):
        if isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            image_data = base64.b64encode(buffer).decode('utf-8')
        elif isinstance(image, str):
            image_path = image

            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        else:
            raise ValueError("Invalid image format")
        
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_input,
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image_data,
                    "mime_type": "image/jpeg",
                },
            ],
        }
        self.message_history.append(message)
        response = self.agent.invoke(self.message_history)
        json_response = json.loads(response.content)
        print("json_response", json_response)
        doors = [door for door in json_response.get("doors", [])]
        if len(doors) == 1:
            answer = json_response.get("answer", "")
            answer += f"{doors[0]['description']}\n"
            answer += "Laisse moi te guider vers la porte trouvé."
            return 0, answer
        if len(doors) >= 2:
            answer = json_response.get("answer", "")
            for idx, door in enumerate(doors):
                answer += f"Porte {idx+1}: {door['description']}\n"
            return 1, answer
        else:
            answer = "Je ne vois pas de portes. Veuillez essayer de vous retourner ou de prendre une autre photo."
            return -1, answer
