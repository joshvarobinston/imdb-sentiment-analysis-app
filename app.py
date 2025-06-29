import joblib
from tensorflow.keras.models import load_model

# Save Tokenizer
joblib.dump(tokenizer, "tokenizer.pkl")

# Save LSTM Model

import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib

# Download NLTK resources


# Load the saved LSTM model and tokenizer
lstm_model = load_model("lstm_sentiment_model.keras")
tokenizer = joblib.load("tokenizer.pkl")

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    words = [word for word in words if word.isalpha() and word not in stop_words]  # Remove stopwords
    return " ".join(words)

# Streamlit UI
st.title("IMDB Sentiment Analysis with LSTM")

# Sidebar menu
option = st.sidebar.radio("Choose Input Type:", ["Single Review", "Upload CSV"])

# **1. Single Review Input**
if option == "Single Review":
    review = st.text_area("Enter a movie review:", "")

    if st.button("Predict Sentiment"):
        if review:
            cleaned_review = preprocess_text(review)

            # Tokenize and pad
            review_seq = tokenizer.texts_to_sequences([cleaned_review])
            review_padded = pad_sequences(review_seq, maxlen=100)

            # Predict sentiment
            lstm_prediction = lstm_model.predict(review_padded)[0][0]
            lstm_result = "Positive" if lstm_prediction >= 0.5 else "Negative"

            # Display result
            st.subheader("Prediction:")
            st.write(f"**LSTM Prediction:** {lstm_result}")
        else:
            st.warning("Please enter a review!")

# **2. Upload CSV for Batch Prediction**
elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with a 'review' column", type=['csv'])

    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)

        if "review" not in new_df.columns:
            st.error("The CSV file must contain a 'review' column.")
        else:
            new_df['cleaned_text'] = new_df['review'].apply(preprocess_text)

            # Tokenize and pad
            X_new_seq = tokenizer.texts_to_sequences(new_df['cleaned_text'])
            X_new_padded = pad_sequences(X_new_seq, maxlen=100)

            # Make predictions
            lstm_predictions = lstm_model.predict(X_new_padded)
            new_df['LSTM Sentiment'] = ["Positive" if p >= 0.5 else "Negative" for p in lstm_predictions]

            st.write("### Predictions:")
            st.write(new_df[['review', 'LSTM Sentiment']])