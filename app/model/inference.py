import torch
from app.model.load_model import load_model, tokenizer, model

def predict(text: str):
    load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence, label_id = torch.max(probs, dim=-1)

    label = model.config.id2label[label_id.item()]

    return {
        "label": label,
        "confidence": round(confidence.item(), 4)
    }
