import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ---------------------------
# Configuration
# ---------------------------
BASE_MODEL_NAME = "distilbert-base-uncased"
ADAPTER_PATH = "./trained_model"  # contains adapter_model.safetensors
MAX_LENGTH = 256

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# ---------------------------
# Load base model + LoRA adapter
# ---------------------------
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=2
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.to(device)
model.eval()

# ---------------------------
# Inference function
# ---------------------------
def predict_sentiment(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)

    label = "Positive" if prediction.item() == 1 else "Negative"

    return {
        "label": label,
        "confidence": round(confidence.item(), 4),
        "device": str(device)
    }
