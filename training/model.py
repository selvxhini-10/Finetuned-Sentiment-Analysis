from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

# Loads pretrained model, applies LoRA adaptation & saves model after training 

MODEL_NAME = "distilbert-base-uncased"

def get_model(num_labels=2):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    return model
