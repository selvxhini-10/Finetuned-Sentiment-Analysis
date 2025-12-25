'''
| Aspect                | LoRA                                                                                               | QLoRA                                                  |
| --------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **Full name**         | Low-Rank Adaptation                                                                                | Quantized LoRA                                         |
| **Memory usage**      | Moderate (adds small rank matrices)                                                                | Low (uses 4-bit quantized model)                       |
| **Training speed**    | Faster than full fine-tuning                                                                       | Even faster, less GPU memory                           |
| **Precision**         | Full precision on added weights                                                                    | Quantized model, may slightly reduce numeric precision |
| **Use case**          | Small/medium models                                                                                | Very large models (e.g., 30B+)                         |
| **Implementation**    | Adds trainable low-rank matrices to existing weights                                                | Combines 4-bit quantization with LoRA                  |

Loads dataset, tokenizer, model with LoRA adaptation, and fine-tunes the model using Hugging Face Trainer
'''
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from tokenizer import get_tokenizer
from model import get_model
import torch

print("CUDA:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0))
print("Using device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def train():
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    train_val = dataset["train"].train_test_split(test_size=0.2, stratify_by_column="label")
    
    train_texts = list(train_val["train"]["text"])
    train_labels = list(train_val["train"]["label"])

    val_texts = list(train_val["test"]["text"])
    val_labels = list(train_val["test"]["label"])

    tokenizer = get_tokenizer()

    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

    # Convert to datasets compatible with Hugging Face Trainer
    import torch
    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)

    model = get_model()

    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        logging_steps=50,
        learning_rate=5e-5,
        fp16=True,
        push_to_hub=False
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

    # Trainer Features: training loop to iterate over dataset, evaluation, saves model periodically at intervals

    trainer.train()
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

if __name__ == "__main__":
    train()
