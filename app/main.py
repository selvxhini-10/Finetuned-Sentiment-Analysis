from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.model.inference import predict

app = FastAPI(title="Sentiment API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(req: PredictRequest):
    return predict(req.text)

@app.get("/")
def root():
    return {"service": "sentiment-api", "status": "running"}
