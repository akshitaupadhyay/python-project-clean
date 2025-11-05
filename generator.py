# generator.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Any 

# 1. MODEL INITIALIZATION (Loads the AI Model)
# This model is specialized for Question Generation
MODEL_NAME = "valhalla/t5-base-qg-hl" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# 2. Set device (CPU or GPU) for better performance
# If a dedicated GPU (cuda) is available, use it; otherwise, use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 3. Define the main function structure that main.py is looking for
# This function is the "brain" of your API
def generate_quiz(text: str) -> Dict[str, List[Dict[str, Any]]]:
    
    # --- The full AI logic would run here ---
    
    # Returning a simple structure to ensure the API starts up and provides a valid response format
    mcqs = [{
        "id": 1,
        "type": "MCQ",
        "question": "Placeholder question from T5 model.",
        "options": ["A", "B", "C", "D"],
        "answer": "A"
    }]
    fill_ins = [{
        "id": 2,
        "type": "Fill-in-the-Blank",
        "question": "The sentence with a blank goes ____________ here.",
        "answer": "The missing word"
    }]

    return {
        "mcqs": mcqs,
        "fill_in_the_blanks": fill_ins
    }