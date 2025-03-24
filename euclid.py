# --------------------------
# This is just a code that uses the basic euclids formula to determine the closest measurement of the candle and not using cnn as of now.
# --------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Data Loading and Preprocessing
# --------------------------

def load_ohlc_data(filepath):
    """
    Load OHLC data from a CSV file.
    Expected columns: timestamp, open, high, low, close.
    """
    data = pd.read_csv(filepath, parse_dates=['timestamp'])
    data.sort_values('timestamp', inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

# --------------------------
# Feature Extraction
# --------------------------

def compute_candle_features(row):
    """
    Compute features for a given candle:
    - Body length: absolute difference between close and open.
    - Upper shadow: high minus the max(open, close).
    - Lower shadow: min(open, close) minus low.
    """
    body = abs(row['close'] - row['open'])
    upper_shadow = row['high'] - max(row['open'], row['close'])
    lower_shadow = min(row['open'], row['close']) - row['low']
    return np.array([body, upper_shadow, lower_shadow])

# --------------------------
# Candle Visualization
# --------------------------

def plot_candle(row):
    """
    Plot the given candle on a 1x1 graph.
    The candle is drawn as a bar (body) with vertical lines (wicks).
    The candle is green if close >= open (bullish), else red (bearish).
    """
    color = 'green' if row['close'] >= row['open'] else 'red'
    
    plt.figure(figsize=(1,1))
    # Plot the body
    body_height = row['close'] - row['open']
    plt.bar(0, body_height, bottom=row['open'], width=0.3, color=color)
    # Plot the wicks (vertical line from low to high)
    plt.vlines(0, row['low'], row['high'], color=color, linewidth=2)
    plt.axis('off')
    plt.title("Simulated Candle")
    plt.show()
    
    # Also return the candle color classification
    return color

# --------------------------
# Similarity Calculation
# --------------------------

def find_most_similar_candle(current_candle, historical_data):
    """
    Compare the current candle to historical candles using Euclidean distance
    between the feature vectors [body, upper_shadow, lower_shadow].
    Returns the most similar historical candle (lowest distance).
    """
    current_features = compute_candle_features(current_candle)
    distances = []
    
    for idx, row in historical_data.iterrows():
        features = compute_candle_features(row)
        distance = np.linalg.norm(current_features - features)  # Euclidean distance
        distances.append(distance)
    
    # Add distances as a column to the historical dataframe for later analysis if needed
    historical_data = historical_data.copy()
    historical_data['distance'] = distances
    most_similar = historical_data.loc[historical_data['distance'].idxmin()]
    return most_similar, historical_data

# --------------------------
# Outcome Evaluation
# --------------------------

def evaluate_historical_outcomes(historical_data, similarity_threshold):
    """
    Identify all historical candles that are similar to the current candle 
    (i.e. have a distance below a defined threshold).
    Then evaluate the outcome of the next candle (buy if next candle's close > current candle's close, sell otherwise).
    Returns the most frequent outcome.
    """
    similar_candles = historical_data[historical_data['distance'] < similarity_threshold]
    outcomes = []
    
    for idx in similar_candles.index:
        # Ensure there's a next candle in the dataset
        if idx < len(historical_data) - 1:
            current = historical_data.loc[idx]
            next_candle = historical_data.loc[idx + 1]
            outcome = 'buy' if next_candle['close'] > current['close'] else 'sell'
            outcomes.append(outcome)
    
    if outcomes:
        most_frequent_outcome = max(set(outcomes), key=outcomes.count)
        return most_frequent_outcome, outcomes
    else:
        return None, outcomes

# --------------------------
# Main Simulation and Analysis
# --------------------------

if __name__ == '__main__':
    # Input parameters:
    # 1. OHLC data for the ETH/USDT pair:
    filepath = 'ETHUSDT_60.csv'  # Replace with the correct path to your dataset.
    ohlc_data = load_ohlc_data(filepath)
    
    # 2. Simulated candle parameters: using the last hourly candle from the data.
    current_candle = ohlc_data.iloc[-1]
    historical_data = ohlc_data.iloc[:-1]  # All candles except the last
    
    # 3. Similarity metric: using Euclidean distance between features [body, upper_shadow, lower_shadow].
    
    # 4. Historical patterns to analyze: outcomes (buy/sell) of the candle following each similar pattern.
    
    # Simulate and plot the current candle
    print("Simulating the current (last hourly) candle...")
    candle_color = plot_candle(current_candle)
    print("Current candle classified as:", candle_color)
    
    # Identify the most similar historical candle
    similar_candle, historical_data_with_distance = find_most_similar_candle(current_candle, historical_data)
    print("\nMost similar historical candle found:")
    print(similar_candle[['timestamp', 'open', 'high', 'low', 'close', 'distance']])
    
    # Optionally, set a threshold for similarity to evaluate multiple occurrences
    similarity_threshold = similar_candle['distance'] * 1.2  # e.g., 20% margin around the closest match
    
    # Evaluate the outcome (buy/sell) following similar candles
    outcome, outcome_list = evaluate_historical_outcomes(historical_data_with_distance, similarity_threshold)
    if outcome:
        print("\nBased on historical patterns, the most frequent outcome following similar candles is:", outcome)
    else:
        print("\nNot enough similar historical patterns were found to determine an outcome.")
