from langchain_openai import ChatOpenAI
from pydantic import BaseModel, model_validator
from typing import List
## constrained int 
from typing import Annotated
import annotated_types
rateInt = Annotated[int, annotated_types.Interval(ge=0, le=10)]
class DoorDescription(BaseModel):
    description: str
    position: str
    DoorOpened: bool
    # @model_validator(mode="after")
    # def validate_position(self):
    #     valid_positions = ["left", "right", "middle"]
    #     print(f"Validating position: {self.position}")
    #     if self.position not in valid_positions:
    #         raise ValueError(f"Position must be one of {valid_positions}")
class DoorDetectionResponse(BaseModel):
    answer: str
    doors: List[DoorDescription]
    ## check if position is in a list of predefined positions
    
def get_agent(t = 0, format = None, name = None):
    params = {"temperature":t,
            "max_tokens":1000,
            "timeout":None,
            "max_retries":2,
            "top_p":0.9,
            }
    if format is None:

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            **params,

        )
    else:
            llm = ChatOpenAI(
            model="gpt-4o-mini",
            **params,
            model_kwargs={"response_format": format} 
            )
    return llm
def run_assistant(query, language_model, t = 0, format = None, name = None):

    params = {"temperature":t,
            "max_tokens":1000,
            "timeout":None,
            "max_retries":2,
            }
    if language_model == "gpt":
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            **params,
            model_kwargs={"response_format": format},
        )
        
    else:
        print("Language model not supported")
        return None

    return llm.invoke(query).content



