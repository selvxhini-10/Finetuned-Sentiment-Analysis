from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Loads saved model and tokenizer -> creates a Hugging Face text-classification pipeline for inference
def load_pipeline():
    model_dir = "./trained_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return nlp_pipeline

nlp_pipeline = load_pipeline()

def predict_sentiment(review: str):
    result = nlp_pipeline(review)
    return result[0]['label']  # POSITIVE / NEGATIVE
