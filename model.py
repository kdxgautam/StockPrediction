# model.py
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from textblob import TextBlob
import streamlit as st

# Fetch real-time stock data from Alpha Vantage (monthly data in this case)
def fetch_stock_data(symbol):
    API_KEY = 'C3E2CQZKSY213XAE'  # Replace with your Alpha Vantage API key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={symbol}&apikey={API_KEY}'
    
    response = requests.get(url)
    data = response.json()
    
    # Check if the data contains 'Monthly Time Series'
    if 'Monthly Time Series' in data:
        time_series = data['Monthly Time Series']
        # Convert the time series data into a pandas DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = df[['4. close']]  # We only need the 'close' price
        df.columns = ['Close']
        
        # Convert 'Close' column to numeric (float)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')  # Convert to float, errors='coerce' will turn invalid values to NaN
        
        df.index = pd.to_datetime(df.index)  # Convert the index to datetime
        df = df.sort_index()  # Sort by date
        
        return df
    else:
        print("Error fetching data. Please check the symbol or try again.")
        return None



def fetch_news(stock_ticker):
    api_key = "91fc8cf73730404fb5b9c38c67038870"  # Replace with your NewsAPI key
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&sortBy=publishedAt&apiKey={api_key}"
    
    response = requests.get(url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get('articles', [])
    else:
        st.error("Failed to fetch news. Check your API key or network.")
        return []
# Preprocess stock data (e.g., use moving averages, lagged features)
def preprocess_data(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()  # Simple Moving Average (50 days)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()  # Simple Moving Average (200 days)
    df['Price_Change'] = df['Close'].pct_change()  # Daily price change percentage
    df['Volatility'] = calculate_volatility(df)  # Add Volatility as a feature
    df = df.dropna()  # Drop rows with NaN values
    return df


# Train a RandomForestRegressor model
def train_model(df):
    features = ['SMA_50', 'SMA_200', 'Price_Change']
    X = df[features]
    y = df['Close']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model

# Predict stock price using the trained model
def predict_stock_price(model, df):
    features = ['SMA_50', 'SMA_200', 'Price_Change']
    X = df[features]
    predictions = model.predict(X)
    return predictions

def analyze_sentiment(article_title):
    analysis = TextBlob(article_title)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive", "green"
    elif polarity == 0:
        return "Neutral", "gray"
    else:
        return "Negative", "red"
def analyze_sentiment_and_recommendation(ticker):
    news_articles = fetch_news(ticker)
    
    sentiments = []
    for article in news_articles[:5]:  # Top 5 news
        sentiment, color = analyze_sentiment(article['content'])  # Use 'content' instead of 'title'
        sentiments.append((sentiment, color))

    return sentiments

def calculate_moving_average(data, short_window=20, long_window=50):
    """
    Calculate short-term and long-term moving averages for the stock data.

    Parameters:
    - data: pandas DataFrame with stock data (must include 'Close' column).
    - short_window: The window size for the short-term moving average.
    - long_window: The window size for the long-term moving average.

    Returns:
    - pandas DataFrame containing moving averages.
    """
    if 'Close' not in data.columns:
        raise ValueError("The input data must contain a 'Close' column.")
    
    moving_averages = pd.DataFrame(index=data.index)
    moving_averages['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    moving_averages['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    return moving_averages

def calculate_volatility(data, window=30):
    """
    Calculate the rolling volatility for the stock data.

    Parameters:
    - data: pandas DataFrame with stock data (must include 'Close' column).
    - window: The rolling window size for calculating volatility.

    Returns:
    - pandas Series containing the volatility values.
    """
    if 'Close' not in data.columns:
        raise ValueError("The input data must contain a 'Close' column.")
    
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    volatility = log_returns.rolling(window=window).std()
    
    return volatility

def analyze_sentiment_and_recommendation(ticker):
    # Fetch the latest news for the stock
    news_articles = fetch_news(ticker)
    
    sentiments = []
    for article in news_articles[:5]:  # Top 5 news
        sentiment, color = analyze_sentiment(article['title'])
        sentiments.append((sentiment, color))

    return sentiments

# Recommendation with confidence based on model performance
def make_recommendation(predicted_price, actual_price, model, processed_data):
    predicted_change = predicted_price - actual_price

    # Calculate model performance (R² score)
    X = processed_data.drop(columns='Close')
    y = processed_data['Close']
    r2_score = model.score(X, y)  # R² score of the model

    recommendation = "Hold"  # Default
    color = "orange"  # Default color

    if r2_score > 0.8:  # High confidence
        if predicted_change > 0:
            recommendation = "Buy"
            color = "green"
        elif predicted_change < 0:
            recommendation = "Sell"
            color = "red"
        else:
            recommendation = "Hold"
            color = "orange"
    else:  # Low confidence
        recommendation = "Hold"
        color = "orange"
    
    return recommendation, color, r2_score






