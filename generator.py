# generator.py
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict, Any 

MODEL_NAME = "valhalla/t5-base-qg-hl" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- UTILITY FUNCTION TO PARSE T5 OUTPUT ---
def parse_t5_output(t5_output: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parses the raw T5 output string into the required JSON structure."""
    
    qa_pairs = t5_output.split("question:")
    mcqs = []
    fill_in_the_blanks = []
    current_id = 1 

    for pair in qa_pairs:
        pair = pair.strip()
        if not pair:
            continue

        if "answer:" in pair:
            q_part, a_part = pair.split("answer:", 1)
            question = q_part.strip()
            answer = a_part.strip()
            
            if len(answer.split()) <= 3:
                fill_in_the_blanks.append({
                    "id": current_id,
                    "type": "Fill-in-the-Blank",
                    "question": question.replace(answer, "____________"),
                    "answer": answer
                })
            else:
                mcqs.append({
                    "id": current_id,
                    "type": "MCQ",
                    "question": question,
                    "options": [answer, "Option B", "Option C", "Option D"], 
                    "answer": answer 
                })
            
            current_id += 1
            
    return {
        "mcqs": mcqs,
        "fill_in_the_blanks": fill_in_the_blanks
    }

# --- MAIN QUESTION GENERATION FUNCTION ---
def generate_quiz(text: str) -> Dict[str, List[Dict[str, Any]]]:
    
    source_text = "generate questions: " + text 
    
    input_ids = tokenizer.encode(
        source_text, 
        return_tensors='pt', 
        max_length=512, 
        truncation=True
    ).to(device)
    
    generated_ids = model.generate(
        input_ids,
        num_beams=4,
        max_length=256,
        early_stopping=True
    )
    
    generated_quiz_text = tokenizer.decode(generated_ids.squeeze(), skip_special_tokens=True)
    
    final_quiz_data = parse_t5_output(generated_quiz_text)
    
    return final_quiz_data