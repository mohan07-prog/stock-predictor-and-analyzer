import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

# --- 1. Get Data ---
# Get 7 years of data for HDFC Life Insurance on the BSE
ticker = "HDFCLIFE.BO"
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=7*365)

print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    print(f"No data found for {ticker}. Please check the ticker symbol.")
else:
    print("Data download complete.")
    
    # --- 2. Preprocess Data ---
    # We only care about the 'Close' price
    close_prices = data[['Close']].values

    # Scale the data (ANNs work best with normalized data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    # Split into training and testing sets (80% train, 20% test)
    train_size = int(len(scaled_prices) * 0.80)
    test_size = len(scaled_prices) - train_size
    train_data = scaled_prices[0:train_size, :]
    test_data = scaled_prices[train_size:len(scaled_prices), :]

    # Helper function to create time-series "look-back" dataset
    # We'll use the past 'look_back' days (X) to predict the next day (y)
    def create_dataset(dataset, look_back=60):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    look_back = 60
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    print(f"Training data shape (X, y): {trainX.shape}, {trainY.shape}")
    print(f"Testing data shape (X, y): {testX.shape}, {testY.shape}")

    # --- 3. Build and Train ANN Model ---
    print("Building and training MLP model...")
    # This is the Multilayer Perceptron (MLP) model
    # We set the learning_rate_init to 0.01 as mentioned in the abstract
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),  # Two hidden layers
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        max_iter=500,
        random_state=42,
        verbose=False
    )

    # Train the model
    model.fit(trainX, trainY)
    print("Model training complete.")

    # --- 4. Make Predictions ---
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    # Invert predictions back to original price scale
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    trainY_orig = scaler.inverse_transform(trainY.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
    testY_orig = scaler.inverse_transform(testY.reshape(-1, 1))

    # --- 5. Evaluate and Plot ---
    # Calculate Mean Squared Error (MSE)
    train_mse = mean_squared_error(trainY_orig, train_predict)
    test_mse = mean_squared_error(testY_orig, test_predict)
    print(f"\nTraining MSE: {train_mse:.2f}")
    print(f"Testing MSE: {test_mse:.2f}")
    
    # Get the dates for the plot
    test_dates = data.index[train_size + look_back + 1 : len(data)]

    # Plot the results
    print("Generating plot...")
    plt.figure(figsize=(16, 8))
    plt.title(f'{ticker} Stock Price Prediction (Mini Project)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (INR)')
    
    # Plot the original 'Close' prices from the test set
    plt.plot(test_dates, testY_orig, label='Actual Price', color='blue', alpha=0.7)
    
    # Plot the model's predicted prices
    plt.plot(test_dates, test_predict, label='Predicted Price', color='red', linestyle='--')
    
    plt.legend()
    plt.grid(True)
    plt.show()