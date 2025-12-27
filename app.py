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