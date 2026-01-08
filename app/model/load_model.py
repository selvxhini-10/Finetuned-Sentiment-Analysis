from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "trained_model"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
