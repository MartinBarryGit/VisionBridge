from utils.llm_assistants import DoorDetectionResponse, get_agent
import numpy as np
import cv2
import json
import base64
class AI_descriptor:
    def __init__(self):
        self.agent = get_agent(format=DoorDetectionResponse)
        self.system_prompt = """You are a live AI vision assistant helping a blind or visually impaired person find a door safely.
        You receive one image per turn (base64). Analyze the current image and conversation context.

        Main goal:
        - Detect doors and guide the user toward a chosen door.

        Response style:
        - Be concise, calm, and actionable (short sentences).
        - Use simple navigation language.
        - Respond in the same language as the user.

        What to report when a door is detected:
        1) Relative direction using a clock reference (12 o'clock = straight ahead, 9 o'clock = full left, 3 o'clock =  full right).
        2) Brief door description (color, material, type if visible).
        3) Handle location (left/right side, approximate height, visibility confidence).

        Multiple doors:
        - Briefly list each door with an index (Door 1, Door 2, ...), clock direction, and one distinguishing detail.
        - Ask the user which door they want.
        - After the user chooses, focus only on that selected door in all following turns.

        If no door is detected:
        - Clearly say no door is visible.
        - Ask the user to turn slightly (left/right), step back if safe, or take another picture.

        When only one target door remains:
        - Confirm: you will guide the user to that door.
        - Provide one immediate next movement instruction.

        Important:
        - Do not invent details that are not visible.
        - If uncertain, say so briefly and give the best safe next action."""
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
if __name__ == "__main__":
    descriptor = AI_descriptor()
    image_path = "/Users/barry/fiftyone/open-images-v7/train/data/ffb70033e95552aa.jpg"
    user_input = "can you describe the doors in this image and tell me where they are?"
    status, description = descriptor.describe_frame(user_input, image_path)
    print("Status:", status)
    print("Description:", description)