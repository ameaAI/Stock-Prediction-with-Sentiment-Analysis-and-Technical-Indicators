# Stock Prediction with Sentiment Analysis and Technical Indicators

This project builds a stock prediction system using real-time financial news sentiment analysis and stock technical indicators. The system processes news articles for sentiment, integrates them with historical stock data, and applies machine learning models to predict stock price movements. AWS services such as Lambda, SageMaker Autopilot, and S3 are leveraged to automate and scale the data pipeline.

## Features
- **Sentiment Analysis**: Fetches real-time news sentiment data using Alpha Vantage's News API.
- **Technical Indicators**: Computes key technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands) from stock price data using Yahoo Finance API.
- **Machine Learning**: Automatically selects, trains, and tunes prediction models using AWS SageMaker Autopilot.
- **Serverless Data Pipeline**: AWS Lambda automates the data fetching and model inference process.

## Prerequisites
- Python 3.x
- `boto3` (AWS SDK for Python)
- `requests` (for API calls)
- `pandas`, `numpy`, `yfinance` (for data handling)
- `AWS Account` for using SageMaker, Lambda, and S3

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ameaAI/Stock-Prediction-with-Sentiment-Analysis-and-Technical-Indicators.git
Install the required Python dependencies:

bash
Copy code
pip install -r requirements.txt
Set up AWS credentials using AWS CLI:

bash
Copy code
aws configure
Workflow
1. Data Collection
Fetches real-time stock news and sentiment scores using Alpha Vantage.
Collects historical stock data and computes technical indicators using Yahoo Finance.
2. Data Preprocessing
Cleans and tokenizes news data for sentiment analysis.
Merges sentiment data with stock price data, creating a feature-rich dataset.
3. Machine Learning
The dataset (sentiment and technical indicators) is fed into AWS SageMaker Autopilot for model selection, training, and tuning.
The best-performing model is used to predict future stock price movements.
4. Prediction & Output
Predictions are saved and can be used to make real-time trading decisions or analyze market sentiment.
AWS Services Used
Lambda: Automates data fetching and model triggering.
S3: Stores data for scalability and easy access.
SageMaker Autopilot: Automates machine learning model selection, training, and tuning.
Example Output
The following sample data is used for predictions:

plaintext
Copy code
Date: 2024-11-29
Title: Amazon Rises 10.4% Since Q3 Earnings: Time to Buy the Stock?
Sentiment: Bullish (Score: 0.498544)
Stock Price (AMZN):
  Open: 205.83
  Close: 207.89
  Volume: 24,892,400
  SMA-20: 204.50
  RSI: 58.68
  Bollinger High: 214.51
  Bollinger Low: 194.50
