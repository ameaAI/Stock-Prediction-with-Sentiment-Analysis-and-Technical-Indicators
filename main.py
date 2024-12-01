import requests
import pandas as pd
from datetime import datetime, timedelta
import boto3
import json
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import os
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
import boto3
import pandas as pd
from io import StringIO

# AWS S3 Client
s3 = boto3.client('s3')

# Alpha Vantage API Key
API_KEY = ''

# S3 Bucket Name
BUCKET_NAME = ""

def generate_time_window():
    """
    Generate dynamic 'time_from' and 'time_to' values for the last two weeks.
    """
    now = datetime.utcnow()

    # Calculate the date two weeks ago from today
    two_weeks_ago = now - timedelta(weeks=2)

    # Set the 'time_from' as 9 AM on the day two weeks ago
    time_from = two_weeks_ago.replace(hour=9, minute=0, second=0, microsecond=0)

    # Set 'time_to' as the current time
    time_to = now

    return {
        "time_from": time_from.strftime('%Y%m%dT%H%M'),
        "time_to": time_to.strftime('%Y%m%dT%H%M')
    }


def fetch_news_from_alphavantage(tickers):
    time_window = generate_time_window()
    tickers_str = ",".join(tickers) 
    
    url = (f"https://www.alphavantage.co/query?"
           f"function=NEWS_SENTIMENT&tickers={tickers_str}&"
           f"time_from={time_window['time_from']}&time_to={time_window['time_to']}&apikey={API_KEY}")
    print("NEWS URL", url)
    try:
        response = requests.get(url, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            
            # Extract the 'feed' field containing news articles
            articles = data.get('feed', [])
            if articles:
                return articles
            else:
                print(f"No news articles found for tickers: {tickers_str}")
                return []
        else:
            print(f"Failed to fetch news: {response.status_code}, {response.text}")
            return []
    
    except Exception as e:
        print(f"Error fetching news for tickers {tickers}: {e}")
        return []


def fetch_stock_prices_with_technical_analysis(stock_symbol, interval="1d", period="6mo"):
    """
    Fetch stock prices and compute technical indicators for a given interval.
    """
    try:
        # Fetch historical data
        stock_data = yf.download(stock_symbol, interval=interval, period=period, group_by="ticker")

        # Flatten MultiIndex columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]

        # Construct the flattened column name for Close prices
        close_column = f"{stock_symbol}_Close"

        # Ensure 'Close' column exists
        if close_column not in stock_data.columns:
            raise KeyError(f"Column '{close_column}' not found in the dataset.")

        # Extract Close prices
        close_prices = stock_data[close_column]

        # Calculate technical indicators
        stock_data['SMA_20'] = SMAIndicator(close_prices, window=20).sma_indicator()
        stock_data['SMA_50'] = SMAIndicator(close_prices, window=50).sma_indicator()
        stock_data['EMA_20'] = EMAIndicator(close_prices, window=20).ema_indicator()
        stock_data['RSI'] = RSIIndicator(close_prices, window=14).rsi()
        macd = MACD(close_prices)
        stock_data['MACD'] = macd.macd()
        stock_data['MACD_Signal'] = macd.macd_signal()
        bb = BollingerBands(close_prices)
        stock_data['Bollinger_High'] = bb.bollinger_hband()
        stock_data['Bollinger_Low'] = bb.bollinger_lband()

        # Reset index for easier merging with news data
        stock_data.reset_index(inplace=True)

        return stock_data
    except Exception as e:
        print(f"Error fetching or analyzing stock prices for {stock_symbol}: {e}")
        return pd.DataFrame()






def consolidate_data_with_technical_analysis(news_data, stock_prices, target_ticker):
    """
    Consolidate Alpha Vantage news data with stock price data and technical indicators.
    """
    consolidated_data = []

    for article in news_data:
        time_published = article.get("time_published", "")
        publish_date = datetime.strptime(time_published, '%Y%m%dT%H%M%S').date()

        # Find stock prices on or before the publish_date
        matching_prices = stock_prices.loc[
            stock_prices['Date'] <= pd.Timestamp(publish_date)
        ].sort_values('Date', ascending=False)  # Closest previous day

        if not matching_prices.empty:
            # Use the closest matching row
            closest_price = matching_prices.iloc[0]

            # Add consolidated data
            consolidated_entry = {
                "date": publish_date,
                "title": article["title"],
                "summary": article.get("summary", ""),
                "overall_sentiment_label": article.get("overall_sentiment_label", "Neutral"),
                "overall_sentiment_score": article.get("overall_sentiment_score", 0.0),
                "source_name": article.get("source", ""),
                "source_domain": article.get("source_domain", ""),
                "topics": ', '.join([topic['topic'] for topic in article.get("topics", [])]),
                "ticker_sentiment": json.dumps(article.get("ticker_sentiment", [])),
                # Use flattened column names
                "price_open": closest_price[f"{target_ticker}_Open"],
                "price_close": closest_price[f"{target_ticker}_Close"],
                "volume": closest_price[f"{target_ticker}_Volume"],
                "SMA_20": closest_price["SMA_20"],
                "SMA_50": closest_price["SMA_50"],
                "EMA_20": closest_price["EMA_20"],
                "RSI": closest_price["RSI"],
                "MACD": closest_price["MACD"],
                "MACD_Signal": closest_price["MACD_Signal"],
                "Bollinger_High": closest_price["Bollinger_High"],
                "Bollinger_Low": closest_price["Bollinger_Low"]
            }

            consolidated_data.append(consolidated_entry)

    return pd.DataFrame(consolidated_data)





def save_to_s3(data, bucket_name, file_name):
    """
    Save data to an S3 bucket as CSV.
    """
    try:
        s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=data,
            ContentType='text/csv'
        )
        print(f"Data saved to S3 bucket: {bucket_name}, file: {file_name}")
    except Exception as e:
        print(f"Error saving data to S3: {e}")



def merge_with_previous_predictions(current_data, s3_bucket, ticker):
    """
    Merge the current dataset with previous predictions from S3.
    """
    import pandas as pd
    import boto3
    from io import StringIO

    s3 = boto3.client('s3')
    try:
        # Fetch the latest predictions file from S3
        latest_file = f"predictions/{ticker}_predictions_latest.csv"
        response = s3.get_object(Bucket=s3_bucket, Key=latest_file)

        # Load previous predictions
        previous_predictions = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))

        # Merge with current data on 'date'
        merged_data = pd.merge(current_data, previous_predictions, on='date', how='left', suffixes=('', '_prev'))
        return merged_data
    except Exception as e:
        print(f"No previous predictions found for {ticker}: {e}")
        return current_data
    
def make_predictions(data, sagemaker_endpoint_name):
    """
    Make predictions using a deployed SageMaker endpoint.
    
    Parameters:
        data (pd.DataFrame): Data to be sent for predictions.
        sagemaker_endpoint_name (str): Name of the deployed SageMaker endpoint.
    
    Returns:
        pd.DataFrame: Original data with predictions added.
    """
    try:
        # Convert the input data to CSV format (required by most models)
        payload = data.to_csv(index=False)
        
        # Initialize SageMaker runtime client
        sm_runtime = boto3.client('sagemaker-runtime')

        # Make predictions
        response = sm_runtime.invoke_endpoint(
            EndpointName=sagemaker_endpoint_name,
            ContentType='text/csv',
            Body=payload
        )

        # Parse the response
        predictions = response['Body'].read().decode('utf-8').strip().split('\n')

        # Add predictions to the original DataFrame
        data['predicted_price'] = pd.to_numeric(predictions)

        return data

    except Exception as e:
        print(f"Error making predictions from SageMaker endpoint: {e}")
        raise

def train_sagemaker_model(train_data, test_data, s3_bucket, s3_prefix, role, output_path, instance_type="ml.m5.xlarge"):
    """
    Train a model in SageMaker and upload training and test data to S3.
    
    Parameters:
        train_data (pd.DataFrame): Training data.
        test_data (pd.DataFrame): Testing/validation data.
        s3_bucket (str): S3 bucket for storing data.
        s3_prefix (str): S3 prefix for data.
        role (str): IAM role for SageMaker.
        output_path (str): S3 path for model output.
        instance_type (str): Instance type for SageMaker training.
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')

        # Save train and test datasets locally
        train_file = "train.csv"
        test_file = "test.csv"
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        # Upload datasets to S3
        s3_train_path = f"s3://{s3_bucket}/{s3_prefix}/train/train.csv"
        s3_test_path = f"s3://{s3_bucket}/{s3_prefix}/test/test.csv"
        s3.upload_file(train_file, s3_bucket, f"{s3_prefix}/train/train.csv")
        s3.upload_file(test_file, s3_bucket, f"{s3_prefix}/test/test.csv")
        print(f"Training data uploaded to: {s3_train_path}")
        print(f"Testing data uploaded to: {s3_test_path}")

        # Define SageMaker estimator
        estimator = Estimator(
            image_uri="683313688378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1",
            role=role,
            instance_count=1,
            instance_type=instance_type,
            output_path=output_path,
            sagemaker_session=boto3.Session()
        )

        # Set hyperparameters
        estimator.set_hyperparameters(
            objective='reg:squarederror',  # For regression
            num_round=100,
            max_depth=5,
            eta=0.2,
            subsample=0.8,
            colsample_bytree=0.8
        )

        # Specify training and validation input
        train_input = TrainingInput(s3_train_path, content_type="text/csv")
        test_input = TrainingInput(s3_test_path, content_type="text/csv")

        # Start training job
        print("Starting SageMaker training job...")
        estimator.fit({"train": train_input, "validation": test_input})

        print(f"Model trained and saved to: {output_path}")

    except Exception as e:
        print(f"Error during SageMaker training: {e}")
        raise

    finally:
        # Clean up local files
        if os.path.exists(train_file):
            os.remove(train_file)
        if os.path.exists(test_file):
            os.remove(test_file)


def split_data(data, train_ratio=0.8, shuffle=True):
    """
    Split a dataset into training and testing subsets.

    Parameters:
        data (pd.DataFrame): The dataset to split.
        train_ratio (float): The proportion of data to use for training (default is 0.8).
        shuffle (bool): Whether to shuffle the data before splitting (default is True).

    Returns:
        tuple: A tuple containing the training set and the testing set as DataFrames.
    """
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset

    train_size = int(len(data) * train_ratio)  # Calculate the number of training samples
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data


def save_predictions_to_s3(predictions, s3_bucket, ticker, current_date):
    """
    Save predictions to an S3 bucket.
    """
 

    s3 = boto3.client('s3')
    filename = f"{ticker}_predictions_{current_date}.csv"
    predictions.to_csv(filename, index=False)

    # Upload to S3
    s3.upload_file(filename, s3_bucket, f"predictions/{filename}")
    print(f"Predictions saved to S3: {s3_bucket}/predictions/{filename}")



if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime

    # Configuration
    s3_bucket = ""  
    ticker = "AMZN"  # Example ticker
    current_date = datetime.utcnow().strftime('%Y-%m-%d')  # Current date

    try:
        # Step 1: Fetch news data
        print(f"Fetching news for {ticker}...")
        news_data = fetch_news_from_alphavantage([ticker])
        if not news_data:
            raise ValueError(f"No news data found for ticker: {ticker}")

        # Step 2: Fetch stock prices with technical analysis
        print(f"Fetching stock prices with technical indicators for {ticker}...")
        stock_prices = fetch_stock_prices_with_technical_analysis(ticker, interval="1d", period="6mo")
        if stock_prices.empty:
            raise ValueError(f"No stock price data available for ticker: {ticker}")

        # Step 3: Merge news and stock data
        print("Consolidating news and stock data...")
        current_data = consolidate_data_with_technical_analysis(news_data, stock_prices, ticker)

       
        current_data.to_csv('data.csv', index=False)

        # Step 4: Merge with previous predictions
        print("Merging with previous predictions...")
        merged_data = merge_with_previous_predictions(current_data, s3_bucket, ticker)

        # Step 5: Split data into training and testing
        print("Splitting data for training and testing...")
        train_data, test_data = split_data(merged_data)

        # Step 6: Train the SageMaker model
        print("Training SageMaker model...")
        train_sagemaker_model(train_data, test_data)

        # Step 7: Make predictions
        print("Making predictions...")
        predictions = make_predictions(merged_data)

        # Step 8: Save predictions to S3
        print("Saving predictions to S3...")
        save_predictions_to_s3(predictions, s3_bucket, ticker, current_date)

        print(f"Pipeline completed successfully for ticker: {ticker}")

    except Exception as e:
        print(f"Error during pipeline execution: {e}")

