import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "trained_model"

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"Model directory not found at: {MODEL_DIR}\n"
        "Ensure you have trained and saved the model."
    )

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    ).to(device)

    model.eval()

    dataset = load_dataset("imdb", split="test")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    loader = DataLoader(dataset, batch_size=16)

    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            preds.extend(batch_preds)
            labels.extend(batch["label"].tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    acc = accuracy_score(labels, preds)

    print("\nEvaluation results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

# ------------------------------------------------------------------
if __name__ == "__main__":
    evaluate()