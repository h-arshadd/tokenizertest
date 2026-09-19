from fastapi import FastAPI
from pydantic import BaseModel, computed_field
import tiktoken

app = FastAPI()

class InputFromUser(BaseModel):
    user_input: str

    @computed_field
    @property
    def token_generator(self) -> list[int]:
        tokenizer = tiktoken.encoding_for_model("gpt-4")
        token_id = tokenizer.encode(self.user_input)
        return token_id

@app.post("/input")
async def input_data(data: InputFromUser):
    return {
        'message': 'Text received.',
        'tokens': data.token_generator
    }

