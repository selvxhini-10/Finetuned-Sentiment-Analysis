from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer
