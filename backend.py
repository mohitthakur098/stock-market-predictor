import os
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
import pathlib
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Prophet import (optional)
Prophet = None
try:
    from prophet import Prophet
    print("✓ Prophet loaded successfully")
except Exception as e:
    print(f"⚠ Prophet not available: {e}")

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backend_debug.log')
    ]
)

print("=" * 60)
print("🇮🇳 Enhanced Indian Stock Predictor Server Starting...")
print("=" * 60)

# ========== ENHANCED TICKER HANDLING FOR INDIAN STOCKS ==========

# Mapping of Indian stock names to their working yfinance symbols
INDIAN_STOCK_MAP = {
    # NSE Stocks (usually work with .NS)
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS', 
    'INFY': 'INFY.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ITC': 'ITC.NS',
    'SBIN': 'SBIN.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'AXISBANK': 'AXISBANK.NS',
    'LT': 'LT.NS',
    'MARUTI': 'MARUTI.NS',
    'TITAN': 'TITAN.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'WIPRO': 'WIPRO.NS',
    'ONGC': 'ONGC.NS',
    'NTPC': 'NTPC.NS',
    'POWERGRID': 'POWERGRID.NS',
    
    # BSE Stocks (fallback with .BO)
    'RELIANCE_BO': 'RELIANCE.BO',
    'TCS_BO': 'TCS.BO',
    'INFY_BO': 'INFY.BO',
}

def get_working_ticker(ticker):
    """Convert input ticker to working yfinance format for Indian stocks"""
    original_ticker = ticker.upper().strip()
    
    # If already has suffix, return as-is
    if '.' in original_ticker:
        return original_ticker
    
    # Try to find in our map
    if original_ticker in INDIAN_STOCK_MAP:
        return INDIAN_STOCK_MAP[original_ticker]
    
    # Default: add .NS suffix for Indian stocks
    return f"{original_ticker}.NS"

def download_indian_stock(ticker, max_retries=3):
    """Enhanced download specifically for Indian stocks on Render"""
    
    working_ticker = get_working_ticker(ticker)
    print(f"📡 Attempting to fetch: {working_ticker}")
    
    for attempt in range(max_retries):
        try:
            # Try different strategies
            if attempt == 0:
                # Strategy 1: Normal download
                df = yf.download(working_ticker, period='1y', interval='1d', 
                                progress=False, threads=False, timeout=10)
            elif attempt == 1:
                # Strategy 2: Shorter period
                df = yf.download(working_ticker, period='6mo', interval='1d',
                                progress=False, threads=False, timeout=10)
            elif attempt == 2:
                # Strategy 3: Try .BO if .NS fails
                if working_ticker.endswith('.NS'):
                    alt_ticker = working_ticker.replace('.NS', '.BO')
                    print(f"🔄 Trying alternative: {alt_ticker}")
                    df = yf.download(alt_ticker, period='6mo', interval='1d',
                                    progress=False, threads=False, timeout=10)
                else:
                    continue
            
            if not df.empty:
                print(f"✅ Success! Got {len(df)} days of data")
                return df
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)[:100]}")
            time.sleep(1)  # Brief pause
    
    # Final attempt with minimal parameters
    try:
        print("🔄 Final attempt with minimal parameters...")
        df = yf.download(working_ticker, period='3mo', progress=False)
        if not df.empty:
            return df
    except:
        pass
    
    return pd.DataFrame()

# ========== HELPER FUNCTIONS ==========

def ensure_1d_series(col):
    if isinstance(col, pd.Series):
        return col
    if isinstance(col, pd.DataFrame):
        if col.shape[1] == 1:
            return col.iloc[:, 0]
        raise TypeError("Expected single-column for Close but got DataFrame with multiple columns.")
    try:
        return pd.Series(col)
    except Exception:
        raise TypeError("Unable to coerce Close to pandas Series.")

def calculate_technical_indicators(df):
    """Calculate RSI, MACD, Bollinger Bands, and Moving Averages"""
    df = df.copy()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # Moving Averages
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    return df

def get_stock_metrics(df):
    """Calculate key stock metrics"""
    close_ser = ensure_1d_series(df['Close'])
    
    # Volatility (annualized)
    returns = close_ser.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100
    
    # Trend (30-day change)
    trend = ((close_ser.iloc[-1] - close_ser.iloc[-30]) / close_ser.iloc[-30]) * 100 if len(close_ser) >= 30 else 0
    
    # Average volume
    if 'Volume' in df.columns:
        avg_volume = df['Volume'].tail(30).mean()
    else:
        avg_volume = None
    
    # 52-week high/low
    high_52w = close_ser.tail(252).max() if len(close_ser) >= 252 else close_ser.max()
    low_52w = close_ser.tail(252).min() if len(close_ser) >= 252 else close_ser.min()
    
    return {
        'volatility': round(volatility, 2),
        'trend_30d': round(trend, 2),
        'avg_volume': int(avg_volume) if avg_volume else None,
        'high_52w': round(high_52w, 2),
        'low_52w': round(low_52w, 2)
    }

# ========== PREDICTION FUNCTIONS ==========

def predict_prophet(df, days):
    if Prophet is None:
        raise RuntimeError("Prophet not installed. Please use another prediction method.")

    df_local = df.copy().sort_index()
    if df_local.empty:
        raise ValueError("Input dataframe is empty.")

    close_ser = ensure_1d_series(df_local['Close'])
    close_ser = pd.to_numeric(close_ser, errors='coerce')
    df_local['Close'] = close_ser
    df_local = df_local.dropna(subset=['Close'])
    
    if len(df_local) < 10:
        raise ValueError("Not enough historical data for Prophet (need at least 10 rows).")

    df_prophet = df_local[['Close']].reset_index().rename(columns={'Date':'ds','Close':'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    m = Prophet(
        daily_seasonality=True, 
        yearly_seasonality=True, 
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    m.fit(df_prophet)

    future = m.make_future_dataframe(periods=days, freq='B')
    forecast = m.predict(future)

    out = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days).copy()
    out['ds'] = pd.to_datetime(out['ds']).dt.strftime('%Y-%m-%d')
    return out.to_dict(orient='records')

def predict_moving_avg(df, days, window=20):
    df_local = df.copy().sort_index()
    close_ser = ensure_1d_series(df_local['Close'])
    close_ser = pd.to_numeric(close_ser, errors='coerce')
    df_local['Close'] = close_ser
    df_local = df_local.dropna(subset=['Close'])

    if len(df_local) < window:
        raise ValueError(f"Not enough data for moving average (need at least {window} rows).")

    last_date = pd.to_datetime(df_local.index[-1])
    ma_value = float(df_local['Close'].tail(window).mean())
    
    recent_change = (df_local['Close'].iloc[-1] - df_local['Close'].iloc[-window]) / window
    
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    forecast_values = [ma_value + (i * recent_change * 0.5) for i in range(days)]
    
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    forecast['yhat_lower'] = forecast['yhat'] * 0.95
    forecast['yhat_upper'] = forecast['yhat'] * 1.05
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.strftime('%Y-%m-%d')
    return forecast.to_dict(orient='records')

def predict_holt_winters(df, days):
    df_local = df.copy().sort_index()
    close_ser = ensure_1d_series(df_local['Close'])
    close_ser = pd.to_numeric(close_ser, errors='coerce')
    df_local['Close'] = close_ser
    df_local = df_local.dropna(subset=['Close'])

    if len(df_local) < 20:
        raise ValueError("Not enough data for Holt-Winters (need at least 20 rows).")

    model = ExponentialSmoothing(
        df_local['Close'], 
        trend='add', 
        seasonal=None, 
        initialization_method='estimated'
    )
    model_fit = model.fit()
    forecast_values = model_fit.forecast(days)
    
    last_date = pd.to_datetime(df_local.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values.values})
    forecast['yhat_lower'] = forecast['yhat'] * 0.95
    forecast['yhat_upper'] = forecast['yhat'] * 1.05
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.strftime('%Y-%m-%d')
    return forecast.to_dict(orient='records')

def predict_linear_regression(df, days):
    df_lr = df.reset_index().sort_values('Date')
    close_ser = ensure_1d_series(df_lr['Close'])
    close_ser = pd.to_numeric(close_ser, errors='coerce')
    df_lr['Close'] = close_ser
    df_lr = df_lr.dropna(subset=['Close'])
    
    if len(df_lr) < 2:
        raise ValueError("Not enough data for linear regression.")
    
    df_lr['t'] = np.arange(len(df_lr))
    model = LinearRegression()
    model.fit(df_lr[['t']], df_lr['Close'])
    
    future_t = np.arange(len(df_lr), len(df_lr) + days)
    forecast_values = model.predict(future_t.reshape(-1, 1))
    
    last_date = pd.to_datetime(df_lr['Date'].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values})
    forecast['yhat_lower'] = forecast['yhat'] * 0.95
    forecast['yhat_upper'] = forecast['yhat'] * 1.05
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.strftime('%Y-%m-%d')
    return forecast.to_dict(orient='records')

def predict_arima(df, days):
    df_local = df.copy().sort_index()
    close_ser = ensure_1d_series(df_local['Close'])
    close_ser = pd.to_numeric(close_ser, errors='coerce')
    df_local['Close'] = close_ser
    df_local = df_local.dropna(subset=['Close'])
    
    if len(df_local) < 30:
        raise ValueError("Not enough historical data for ARIMA (need at least 30 rows).")
    
    model = ARIMA(df_local['Close'], order=(2, 1, 2))
    model_fit = model.fit()
    forecast_values = model_fit.forecast(steps=days)
    
    last_date = pd.to_datetime(df_local.index[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': forecast_values.values})
    forecast['yhat_lower'] = forecast['yhat'] * 0.95
    forecast['yhat_upper'] = forecast['yhat'] * 1.05
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.strftime('%Y-%m-%d')
    return forecast.to_dict(orient='records')

def predict_random_forest(df, days):
    """Random Forest prediction using technical features"""
    df_local = df.copy().sort_index()
    close_ser = ensure_1d_series(df_local['Close'])
    close_ser = pd.to_numeric(close_ser, errors='coerce')
    df_local['Close'] = close_ser
    df_local = df_local.dropna(subset=['Close'])
    
    if len(df_local) < 50:
        raise ValueError("Not enough data for Random Forest (need at least 50 rows).")
    
    # Create features
    df_local['MA5'] = df_local['Close'].rolling(5).mean()
    df_local['MA20'] = df_local['Close'].rolling(20).mean()
    df_local['Volatility'] = df_local['Close'].rolling(10).std()
    df_local['Returns'] = df_local['Close'].pct_change()
    df_local = df_local.dropna()
    
    # Prepare training data
    features = ['MA5', 'MA20', 'Volatility', 'Returns']
    X = df_local[features].values[:-1]
    y = df_local['Close'].values[1:]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict future
    last_date = pd.to_datetime(df_local.index[-1])
    predictions = []
    current_features = df_local[features].iloc[-1].values.reshape(1, -1)
    
    for _ in range(days):
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        # Update features (simplified)
        current_features = np.roll(current_features, -1)
        current_features[0, -1] = (pred - predictions[-1] if len(predictions) > 1 else 0) / (predictions[-1] if predictions[-1] != 0 else 1)
    
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days)
    forecast = pd.DataFrame({'ds': future_dates, 'yhat': predictions})
    forecast['yhat_lower'] = forecast['yhat'] * 0.92
    forecast['yhat_upper'] = forecast['yhat'] * 1.08
    forecast['ds'] = pd.to_datetime(forecast['ds']).dt.strftime('%Y-%m-%d')
    return forecast.to_dict(orient='records')

# ========== ROUTES ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'server_time': datetime.now().isoformat(),
        'prophet_available': Prophet is not None,
        'indian_stocks_supported': list(INDIAN_STOCK_MAP.keys())[:10]
    })

@app.route('/predict')
def predict():
    """Main prediction endpoint for Indian stocks"""
    ticker = request.args.get('ticker', '').strip()
    if not ticker:
        return jsonify({
            'error': "Missing 'ticker' parameter",
            'example': "/predict?ticker=RELIANCE&future_days=30&strategy=moving_avg",
            'supported_stocks': list(INDIAN_STOCK_MAP.keys())
        }), 400

    try:
        future_days = int(request.args.get('future_days', 30))
    except ValueError:
        return jsonify({'error': "'future_days' must be an integer"}), 400
    
    if future_days < 1 or future_days > 90:
        return jsonify({'error': "'future_days' must be between 1 and 90"}), 400

    strategy = request.args.get('strategy', 'moving_avg')
    
    if strategy == 'prophet' and Prophet is None:
        return jsonify({
            'error': 'Prophet is not available. Please use another prediction method.',
            'available_methods': ['moving_avg', 'linear_regression', 'holt_winters', 'arima', 'random_forest']
        }), 400

    print(f"\n📈 Processing: {ticker.upper()} | Strategy: {strategy} | Days: {future_days}")

    # Use enhanced download for Indian stocks
    df = download_indian_stock(ticker)
    
    if df.empty:
        return jsonify({
            'error': f"No data found for '{ticker}' on Render servers.",
            'suggestion': f"Try: {get_working_ticker(ticker)}",
            'working_examples': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ITC'],
            'tip': 'Use stock names without .NS/.BO suffix (e.g., RELIANCE not RELIANCE.NS)'
        }), 404

    print(f"✅ Downloaded {len(df)} trading days of data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    if 'Close' not in df.columns:
        return jsonify({'error': f"No 'Close' price data for {ticker}"}), 404

    df = df.sort_index()
    df.index.name = 'Date'
    df = df.dropna(subset=['Close'])

    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(df)
    
    # Get stock metrics
    metrics = get_stock_metrics(df)

    try:
        last_close = float(df['Close'].iloc[-1])
        print(f"💰 Last close: ₹{last_close:.2f}")
    except Exception:
        last_close = None

    # Prepare history with technical indicators
    try:
        history_df = df_with_indicators[['Close', 'RSI', 'MACD', 'MA20', 'MA50']].tail(120)
        history_df.index = pd.to_datetime(history_df.index)
        
        history = []
        for idx, row in history_df.iterrows():
            history.append({
                'ds': idx.strftime('%Y-%m-%d'),
                'close': float(row['Close']),
                'rsi': float(row['RSI']) if pd.notna(row['RSI']) else None,
                'macd': float(row['MACD']) if pd.notna(row['MACD']) else None,
                'ma20': float(row['MA20']) if pd.notna(row['MA20']) else None,
                'ma50': float(row['MA50']) if pd.notna(row['MA50']) else None
            })
    except Exception as e:
        print(f"⚠ Error preparing history: {str(e)}")
        history = []

    print(f"🎯 Running {strategy} prediction...")

    try:
        if strategy == 'prophet':
            forecast = predict_prophet(df, future_days)
        elif strategy == 'moving_avg':
            forecast = predict_moving_avg(df, future_days)
        elif strategy == 'holt_winters':
            forecast = predict_holt_winters(df, future_days)
        elif strategy == 'linear_regression':
            forecast = predict_linear_regression(df, future_days)
        elif strategy == 'arima':
            forecast = predict_arima(df, future_days)
        elif strategy == 'random_forest':
            forecast = predict_random_forest(df, future_days)
        else:
            return jsonify({'error': 'Invalid strategy'}), 400

        print(f"✅ Prediction completed successfully!\n")

        return jsonify({
            'ticker': ticker.upper(),
            'last_close': last_close,
            'history': history,
            'forecast': forecast,
            'metrics': metrics,
            'data_points': len(df),
            'period': f"{df.index[0].date()} to {df.index[-1].date()}"
        })
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}")
        logging.error(f"Prediction failed for {ticker}: {str(e)}")
        
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'suggestion': 'Try a different prediction method or fewer days'
        }), 500

@app.route('/compare')
def compare():
    """Compare multiple Indian stocks"""
    tickers = request.args.get('tickers', '').split(',')
    tickers = [t.strip() for t in tickers if t.strip()]
    
    if not tickers or len(tickers) > 5:
        return jsonify({'error': 'Please provide 1-5 Indian stock symbols'}), 400
    
    results = {}
    for ticker in tickers:
        try:
            df = download_indian_stock(ticker)
            if df.empty:
                continue
                
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df = df.sort_index()
            close_ser = ensure_1d_series(df['Close'])
            
            # Calculate returns
            start_price = float(close_ser.iloc[0])
            end_price = float(close_ser.iloc[-1])
            returns = ((end_price - start_price) / start_price) * 100
            
            # Get metrics
            metrics = get_stock_metrics(df)
            
            results[ticker.upper()] = {
                'returns_1y': round(returns, 2),
                'current_price': round(end_price, 2),
                'volatility': metrics['volatility'],
                'trend_30d': metrics['trend_30d'],
                'high_52w': metrics['high_52w'],
                'low_52w': metrics['low_52w']
            }
        except Exception as e:
            print(f"⚠ Error comparing {ticker}: {str(e)}")
            continue
    
    return jsonify(results)

@app.route('/supported-stocks')
def supported_stocks():
    """List all supported Indian stocks"""
    return jsonify({
        'indian_stocks': INDIAN_STOCK_MAP,
        'total': len(INDIAN_STOCK_MAP),
        'tip': 'Use stock names without suffix (e.g., RELIANCE not RELIANCE.NS)'
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    
    print(f"\n✅ Server ready on port {port}")
    print(f"🌐 Access at: http://0.0.0.0:{port}")
    print(f"📊 Supported Indian Stocks: {len(INDIAN_STOCK_MAP)}")
    print(f"🤖 Prophet available: {'Yes' if Prophet else 'No'}\n")
    
    app.run(debug=False, host='0.0.0.0', port=port)
