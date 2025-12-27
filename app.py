import streamlit as st
from inference.predict import predict_sentiment

# Custom HTML/CSS for the banner
custom_html = """
<div class="banner">
    <img src="https://img.pikbest.com/backgrounds/20200504/technology-blue-minimalist-banner-background_2754199.jpg!bwr800" alt="Banner Image">
</div>
<style>
    .banner {
        width: 100%;
        height: 200px;
        overflow: hidden;
    }
    .banner img {
        width: 100%;
        object-fit: cover;
    }
</style>
"""
# Display the custom HTML
st.markdown(custom_html, unsafe_allow_html=True)

# Inject CSS to change the font
st.markdown(
    """
    <style>
        /* Change font for sidebar */
        .css-1d391kg, .css-1v3fvcr {  /* Adjusts sidebar text */
            font-family: 'Source Sans Pro', sans-serif !important;
        }
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="Sentiment Analyzer")

st.title("Fine-Tuned Sentiment Analysis with LoRA and Transformers")

st.write("This project implements a movie review sentiment analysis system using a fine-tuned DistilBERT transformer model. The model is trained and evaluated on the IMDb dataset with Hugging Face’s Transformers library and accelerated with CUDA on an NVIDIA GeForce RTX 4060 Laptop GPU. Fine-tuning is performed using Low-Rank Adaptation (LoRA) with Parameter-Efficient Fine-Tuning (PEFT), integrated into Hugging Face’s Trainer API. Inference is deployed through a Streamlit web application displaying predicted sentiment labels (positive/negative), confidence scores and runtime device information (CPU/GPU).")

st.subheader("Try it out!")
st.write("Enter a movie review below to analyze its sentiment. The model will predict whether the review is positive or negative along with a confidence score.")

text = st.text_area("Enter a movie review")

if st.button("Analyze"):
    if text.strip():
        with st.spinner("Running inference..."):
            result = predict_sentiment(text)
        if result['label'] == "Positive":
            st.success(f"Prediction: **{result['label']}**")
        else:
            st.error(f"Prediction: **{result['label']}**")
        st.progress(result["confidence"])
        st.write(f"Confidence: `{result['confidence'] * 100:.2f}%`")
        st.write(f"Processed on: `{result['device']}`")
    else:
        st.warning("Please enter a review.")

st.markdown("---")

st.header("How does Fine-Tuning Work?")
st.write("Large Language Models (LLMs) are pre-trained on massive, generalized datasets to learn broad linguistic patterns. This generalization often lacks the specialization required for domain-specific applications such as sentiment analysis, customer support automation or legal document review. Fine-tuning adapts a pre-trained model to a specific task by training it on a smaller, task-specific dataset. Traditional full fine-tuning updates all model parameters, which can be computationally expensive and time consuming, often demanding hours or days of training and large amounts of GPU memory (VRAM). ")

st.header("Parameter-Efficient Fine-Tuning with LoRA")
st.write("To address these limitations, this project uses LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning (PEFT) technique. Instead of updating all model weights, PEFT and LoRA adds small, trainable low-rank matrices into selected attention layers of the transformer. In standard fine-tuning, updating an 𝑁×𝑁 weight matrix requires modifying all 𝑁^2 parameters. LoRA decomposes this update into two smaller matrices with rank 𝑟, significantly reducing the number of trainable parameters.")

st.subheader("What was Actually Trained")
st.write("The base DistilBERT model weights were not fine-tuned directly and remained frozen. Only the LoRA adapter parameters were trained, which were stored separately from the base model. During inference, the adapters are dynamically applied on top of the frozen base model.")

st.subheader("QLoRA (conceptual extension)")
st.write("Quantized Low-Rank Adaptation (QLoRA) takes this a step further by combining parameter-efficient fine-tuning with model quantization. In QLoRA, the pre-trained model weights are loaded in a quantized format, reducing memory consumption. This makes it possible for fine-tuning extremely large models on consumer GPUs.")

st.header("Model Performance Evaluation Workflow")
st.write("After fine-tuning the DistilBERT sentiment classification model, an evaluation pipeline objectively measures its performance on unseen data. Evaluation was conducted using the IMDb test split, ensuring that the model was assessed on reviews it had not encountered during training. ")

st.subheader("Performance Metrics")

option = st.selectbox("Select Metric to View Definitions",
    ('Accuracy', 'Precision', 'Recall', 'F1 Score'),
    index=0  # sets the default value to the first option
)
## display all metrics in coloured box 
if option == 'Accuracy':
    st.write("0.8919 -> Measures the proportion of correctly classified reviews across the entire test set")
elif option == 'Precision':
    st.write("0.8824 -> Quantifies how many of the reviews predicted as positive are truly positive (false positives).")
elif option == 'Recall':
    st.write("0.9044 -> Quantifies how many of the reviews predicted as positive are truly positive (false negatives).")
else:  # F1 Score
    st.write("0.8933 -> Represents the harmonic mean of precision and recall for a balanced measure of overall classification quality")

st.success("These results show that the model generalizes well to unseen data and achieves high recall while maintaining strong precision. ")