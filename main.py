# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from generator import generate_quiz 

app = FastAPI()

class TextInput(BaseModel):
    # INDENTATION HERE
    text: str 
    
@app.post("/generate-quiz")
def create_quiz(input: TextInput):
    # INDENTATION HERE
    quiz_output = generate_quiz(input.text)
    # INDENTATION HERE
    return {
        # INDENTATION HERE
        "source_text_used": input.text,
        "generated_quiz": quiz_output
    }
    
@app.get("/")
def read_root():
    # INDENTATION HERE
    return {"status": "AI Study Buddy API is running!"}