# IMDB Sentiment Analysis Web App 🎬

This is a Streamlit web app that performs sentiment analysis on IMDB movie reviews using an LSTM model.

## Features
- Single review sentiment prediction
- Batch prediction via CSV upload

## Files
- `app.py`: Streamlit app code
- `lstm_sentiment_model.keras`: Pretrained LSTM model
- `tokenizer.pkl`: Tokenizer used during model training
- `IMDB.csv`: Dataset (optional for demo)
- `customer_supportchatpot1.ipynb`: Model training notebook

## How to Run Locally
1. Clone the repo :   
 ```bash
   git clone https://github.com/joshvarobinston/imdb-sentiment-analysis-app.git
   cd imdb-sentiment-analysis-app
2. Install requirements: pip install -r requirements.txt
3. Run the app: streamlit run app.py


