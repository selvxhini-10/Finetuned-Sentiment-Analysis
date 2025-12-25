import streamlit as st
from inference.predict import predict_sentiment

st.title("Movie Review Sentiment Analyzer")
review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review.strip():
        prediction = predict_sentiment(review)
        st.success(f"Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter a review.")
