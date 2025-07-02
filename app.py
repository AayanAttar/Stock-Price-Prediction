import os
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the pre-trained model
try:
    model = joblib.load('hubble_model.pkl')
except:
    model = None
    print("Warning: Could not load the model. Using dummy predictions.")

def get_stock_data(ticker, period='1y'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(df):
    """Prepare data for prediction with all required features (39 features)"""
    try:
        # Create a copy of the dataframe
        df = df[['Close']].copy()
        
        # 1. Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log1p(df['Returns'])
        
        # 2. Moving Averages (7 features)
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # 3. Exponential Moving Averages (4 features)
        for window in [5, 10, 20, 50]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # 4. Volatility features (5 features)
        for window in [5, 10, 20, 30, 60]:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        # 5. Bollinger Bands (3 features)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = (
            df['Close'].rolling(window=20).mean() + 2*df['Close'].rolling(window=20).std(),
            df['Close'].rolling(window=20).mean(),
            df['Close'].rolling(window=20).mean() - 2*df['Close'].rolling(window=20).std()
        )
        
        # 6. RSI (1 feature)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 7. MACD (2 features)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 8. Momentum features (5 features)
        for days in [1, 3, 5, 10, 20]:
            df[f'Momentum_{days}'] = df['Close'] - df['Close'].shift(days)
        
        # 9. Price Rate of Change (5 features)
        for days in [1, 3, 5, 10, 20]:
            df[f'ROC_{days}'] = (df['Close'] - df['Close'].shift(days)) / df['Close'].shift(days)
        
        # 10. Additional technical indicators (4 features)
        # ATR (Average True Range)
        high_low = df['Close'].diff()
        high_close = (df['Close'] - df['Close'].shift(1)).abs()
        low_close = (df['Close'] - df['Close'].shift(1)).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # OBV (On-Balance Volume)
        # Note: We don't have volume data, so we'll skip this one
        
        # Remove any rows with NaN values that were created by indicators
        df = df.dropna()
        
        # Get all features except the target variable and any remaining non-numeric columns
        feature_columns = [col for col in df.columns if col != 'Close' and pd.api.types.is_numeric_dtype(df[col])]
        
        # Ensure we have exactly 39 features (pad with zeros if needed)
        if len(feature_columns) < 39:
            # Add zero columns for missing features
            for i in range(len(feature_columns), 39):
                col_name = f'feature_{i}'
                df[col_name] = 0
                feature_columns.append(col_name)
        elif len(feature_columns) > 39:
            # Take only the first 39 features
            feature_columns = feature_columns[:39]
        
        X = df[feature_columns].values
        
        # Scale the features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Return the most recent data point
        return X_scaled[-1:], scaler
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None

def create_plot(df, prediction):
    """Create interactive plot with Plotly"""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4')
    ))
    
    # Add prediction point
    last_date = df.index[-1]
    next_date = last_date + timedelta(days=1)
    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[prediction],
        mode='markers',
        name='Prediction',
        marker=dict(color='red', size=10)
    ))
    
    # Update layout
    fig.update_layout(
        title='Stock Price History and Prediction',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_white',
        showlegend=True
    )
    
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        form_data = request.form
        ticker = form_data.get('ticker', 'AAPL').upper()
        stock_volume = float(form_data.get('stock_volume', 0))
        
        # Get stock data
        df = get_stock_data(ticker)
        
        if df is not None and not df.empty:
            # Add the stock volume to the dataframe
            df['Volume'] = stock_volume
            
            # Get optional market data
            market_columns = [
                'Natural_Gas_Price', 'Natural_Gas_Vol', 'Crude_oil_Price', 'Crude_oil_Vol',
                'Copper_Price', 'Copper_Vol', 'Bitcoin_Price', 'Bitcoin_Vol',
                'Platinum_Price', 'Platinum_Vol', 'Ethereum_Price', 'Ethereum_Vol',
                'S&P_500_Price', 'Nasdaq_100_Price', 'Nasdaq_10_Vol', 'Apple_Price',
                'Apple_Vol', 'Tesla_Price', 'Tesla_Vol', 'Microsoft_Price',
                'Microsoft_Vol', 'Silver_Price', 'Silver_Vol', 'Google_Price',
                'Google_Vol', 'Nvidia_Price', 'Nvidia_Vol', 'Berkshire_Price',
                'Berkshire_Vol', 'Netflix_Price', 'Netflix_Vol', 'Amazon_Price',
                'Amazon_Vol', 'Meta_Price', 'Meta_Vol', 'Gold_Price', 'Gold_Vol'
            ]
            
            # Add market data to the dataframe (default to 0 if not provided)
            for col in market_columns:
                df[col] = float(form_data.get(col, 0))
            
            X, scaler = prepare_data(df)
            
            if X is not None and model is not None:
                # Make prediction - ensure X is 2D array
                X_pred = X.reshape(1, -1) if len(X.shape) == 1 else X[-1:]
                prediction = float(model.predict(X_pred)[0])
                
                # Get the current price for display
                current_price = float(df['Close'].iloc[-1])
                
                # Create plot for web interface
                plot_div = create_plot(df, prediction)
                
                return render_template('index.html', 
                                    prediction=prediction, 
                                    ticker=ticker,
                                    plot_div=plot_div,
                                    current_price=current_price,
                                    form_data=form_data)
        
        # If we get here, something went wrong
        return render_template('index.html', 
                             error="Could not fetch data or make prediction. Please check the ticker symbol and try again.",
                             form_data=form_data)
    
    # For GET requests, just show the form with empty values
    return render_template('index.html', form_data={})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL').upper()
    
    # Get stock data
    df = get_stock_data(ticker)
    
    if df is not None and not df.empty:
        # Prepare data for prediction
        X, scaler = prepare_data(df)
        
        if X is not None and model is not None:
            # Make prediction - ensure X is 2D array
            X_pred = X.reshape(1, -1) if len(X.shape) == 1 else X[-1:]
            prediction = float(model.predict(X_pred)[0])
            
            # Get the current price for display
            current_price = float(df['Close'].iloc[-1])
            
            return jsonify({
                'status': 'success',
                'ticker': ticker,
                'prediction': round(prediction, 2),
                'current_price': round(current_price, 2)
            })
    
    return jsonify({
        'status': 'error',
        'message': 'Could not make prediction. Please try again.'
    }), 400




if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static', exist_ok=True)
    
    # Run the app
    app.run(debug=True, port=5000)
