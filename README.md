# Fine-Tuned Sentiment Analysis with LoRA & Transformers

This project implements a **GPU-accelerated sentiment analysis system** for movie reviews using a **fine-tuned DistilBERT transformer model**. The model is adapted using **Low-Rank Adaptation (LoRA)** under the **Parameter-Efficient Fine-Tuning (PEFT)** framework and trained on the **IMDb dataset**.

The system supports **training, evaluation, and real-time inference**, with predictions served through an interactive **Streamlit web application** that displays sentiment labels, confidence scores, and runtime device information (CPU/GPU).

---

## Key Features

* **Transformer-based sentiment classification** using DistilBERT
* **Parameter-efficient fine-tuning** with LoRA (PEFT)
* **CUDA-enabled training and inference** (NVIDIA RTX 4060 Laptop GPU)
* **Separated training, evaluation, and inference pipelines**
* **Streamlit UI** for interactive sentiment prediction
* **Confidence score visualization** for predictions
* **Model performance evaluation** (Accuracy, Precision, Recall, F1)

---

## Model Architecture

* **Base Model:** `distilbert-base-uncased`
* **Task:** Binary sentiment classification (Positive / Negative)
* **Fine-Tuning Method:** LoRA (Low-Rank Adaptation)
* **Training Framework:** Hugging Face `Trainer` API
* **Adapters:** Stored separately from the frozen base model

### Why LoRA?

Traditional full fine-tuning updates all model parameters, which is computationally expensive and memory-intensive. LoRA freezes the base model and trains only small low-rank adapter matrices injected into attention layers. This significantly reduces:

* GPU memory usage
* Training time
* Storage requirements

while maintaining competitive performance.

---

## Project Structure

```
Finetuned-Sentiment-Analysis/
│
├── training/
│   ├── train.py          # Model fine-tuning with Hugging Face Trainer
│   └── model.py          # DistilBERT + LoRA adapter configuration
│
├── evaluation/
│   └── evaluate.py       # Offline evaluation on IMDb test set
│
├── inference/
│   └── predict.py        # Model + adapter loading and inference logic
│
├── trained_model/
│   ├── adapter_model.safetensors  # Trained LoRA adapter weights
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── README.md
│
├── results/
│   └── checkpoint-*      # Training checkpoints
│
├── app.py                # Streamlit frontend
├── requirements.txt
└── README.md
```

---

## Training

The model is fine-tuned using the IMDb dataset via Hugging Face’s `Trainer` API. Only the LoRA adapter parameters are updated during training.

```bash
python training/train.py
```

Training artifacts (checkpoints, logs) are saved in the `results/` directory, while the final adapter and tokenizer are stored in `trained_model/`.

---

## Evaluation

Model performance is evaluated on the IMDb test split using standard classification metrics.

```bash
python evaluation/evaluate.py
```

### Metrics Reported

* Accuracy
* Precision
* Recall
* F1 Score

Example results:

```
Accuracy:  0.8919
Precision: 0.8824
Recall:    0.9044
F1 Score:  0.8933
```

---

## Inference & Streamlit App

The Streamlit application loads the frozen base model and applies the trained LoRA adapter for inference.

```bash
streamlit run app.py
```

### UI Features

* Sentiment prediction (Positive / Negative)
* Confidence score visualization
* Runtime device display (CPU vs GPU)
* Real-time text input for inference

---

## Hardware & Environment

* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
* **CUDA:** Enabled
* **Python:** 3.11
* **PyTorch:** CUDA-enabled build
* **OS:** Windows

---

## Technologies Used

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT (LoRA)
* CUDA
* Streamlit
* scikit-learn