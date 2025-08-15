import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import time
import google.generativeai as genai
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="Crypto Trading Platform",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Expanded cryptocurrency list
CRYPTO_LIST = [
    'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC',
    'LINK', 'UNI', 'ATOM', 'LTC', 'BCH', 'ICP', 'FIL', 'TRX', 'ETC', 'XLM',
    'VET', 'ALGO', 'HBAR', 'MANA', 'SAND', 'CRV', 'AAVE', 'MKR', 'COMP', 'SNX',
    'SUSHI', '1INCH', 'YFI', 'BAT', 'ZEC', 'DASH', 'NEO', 'QTUM', 'ONT', 'ZIL'
]

# Initialize session state with expanded portfolio
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {crypto: 0.0 for crypto in CRYPTO_LIST}
if 'cash_balance' not in st.session_state:
    st.session_state.cash_balance = 10000.0  # Starting with $10,000
if 'trading_history' not in st.session_state:
    st.session_state.trading_history = []
if 'futures_positions' not in st.session_state:
    st.session_state.futures_positions = []
if 'staking_positions' not in st.session_state:
    st.session_state.staking_positions = []

class CryptoDataAPI:
    """Handles cryptocurrency data from multiple sources"""

    @staticmethod
    def get_coinbase_data(symbol: str) -> Dict:
        """Get data from Coinbase API"""
        try:
            url = f"https://api.coinbase.com/v2/exchange-rates?currency={symbol}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}

    @staticmethod
    def get_binance_data(symbol: str) -> Dict:
        """Get data from Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return {}

    @staticmethod
    def get_yfinance_data(symbol: str, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """Get data from Yahoo Finance with custom interval"""
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            # For 1-minute data, we need to limit the period
            if interval == "1m":
                if period in ["1d", "2d", "5d", "7d"]:
                    data = ticker.history(period=period, interval=interval)
                else:
                    # For longer periods with 1min interval, use max 7 days
                    data = ticker.history(period="7d", interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_binance_klines(symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
        """Get kline/candlestick data from Binance API"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'limit': limit
            }
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
                ])

                # Convert to proper types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)

                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                return df
        except Exception as e:
            print(f"Error fetching Binance klines for {symbol}: {e}")
            return pd.DataFrame()

class GeminiAIAgent:
    """AI Agent using Google Gemini for crypto analysis"""

    def __init__(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

    def analyze_market(self, crypto_data: Dict) -> str:
        """Analyze market data using Gemini AI"""
        if not self.model:
            return "AI analysis requires Gemini API key"

        try:
            prompt = f"""
            Analyze the following cryptocurrency market data and provide insights:

            {json.dumps(crypto_data, indent=2)}

            Please provide:
            1. Market trend analysis
            2. Price predictions (short-term)
            3. Risk assessment
            4. Trading recommendations
            5. Technical indicators analysis

            Keep the analysis concise and actionable.
            """

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI analysis error: {str(e)}"

    def get_trading_advice(self, symbol: str, current_price: float, portfolio: Dict) -> str:
        """Get AI trading advice for specific crypto"""
        if not self.model:
            return "AI advice requires Gemini API key"

        try:
            prompt = f"""
            Provide trading advice for {symbol} at current price ${current_price:.2f}.
            Current portfolio: {portfolio}

            Consider:
            1. Current market conditions
            2. Portfolio diversification
            3. Risk management
            4. Entry/exit points

            Provide specific buy/sell/hold recommendations with reasoning.
            """

            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"AI advice error: {str(e)}"

def create_price_chart(data: pd.DataFrame, title: str, chart_type: str = "candlestick") -> go.Figure:
    """Create interactive price chart with multiple types"""
    fig = go.Figure()

    if not data.empty:
        if chart_type == "candlestick":
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ))
        elif chart_type == "line":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#00aaff', width=2)
            ))
        elif chart_type == "area":
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                fill='tonexty',
                mode='lines',
                name='Price',
                line=dict(color='#00aaff'),
                fillcolor='rgba(0, 170, 255, 0.3)'
            ))

        # Add volume subplot
        if 'Volume' in data.columns:
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                yaxis='y2',
                opacity=0.3,
                marker_color='#888888'
            ))

    fig.update_layout(
        title=title,
        yaxis_title="Price (USD)",
        xaxis_title="Time",
        template="plotly_dark",
        height=600,
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False
    )

    return fig

def simulate_trading(action: str, symbol: str, amount: float, price: float):
    """Simulate trading operations"""
    if action == "BUY":
        cost = amount * price
        if cost <= st.session_state.cash_balance:
            st.session_state.cash_balance -= cost
            st.session_state.portfolio[symbol] += amount
            st.session_state.trading_history.append({
                'timestamp': datetime.now(),
                'action': 'BUY',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'total': cost
            })
            return True, f"Bought {amount} {symbol} at ${price:.2f}"
        else:
            return False, "Insufficient funds"

    elif action == "SELL":
        if st.session_state.portfolio[symbol] >= amount:
            revenue = amount * price
            st.session_state.cash_balance += revenue
            st.session_state.portfolio[symbol] -= amount
            st.session_state.trading_history.append({
                'timestamp': datetime.now(),
                'action': 'SELL',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'total': revenue
            })
            return True, f"Sold {amount} {symbol} at ${price:.2f}"
        else:
            return False, "Insufficient holdings"

def main():
    st.title("🚀 Advanced Crypto Trading Platform")
    st.markdown("*AI-Powered Trading with Multiple Exchange Integration*")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Markets", "Buy Crypto", "Trading", "Futures", "Earn", "AI Analysis", "Portfolio"]
    )

    # API Configuration
    with st.sidebar.expander("API Configuration"):
        gemini_api_key = st.text_input("Gemini API Key", type="password")
        if gemini_api_key:
            st.success("✅ Gemini AI connected")

    # Initialize AI Agent
    ai_agent = GeminiAIAgent(gemini_api_key)

    # Main content based on selected page
    if page == "Markets":
        st.header("📊 Cryptocurrency Markets")

        # Market overview - Top cryptocurrencies
        st.subheader("Top Cryptocurrencies")

        # Create tabs for different market views
        tab1, tab2, tab3 = st.tabs(["Market Overview", "Price Charts", "Market Screener"])

        with tab1:
            # Display top cryptos in a grid
            cols = st.columns(5)
            top_cryptos = CRYPTO_LIST[:20]  # Show top 20

            prices = {}
            market_data = []

            for i, crypto in enumerate(top_cryptos):
                with cols[i % 5]:
                    # Get price data
                    binance_data = CryptoDataAPI.get_binance_data(crypto)
                    if binance_data:
                        price = float(binance_data.get('lastPrice', 0))
                        change = float(binance_data.get('priceChangePercent', 0))
                        volume = float(binance_data.get('volume', 0))
                        prices[crypto] = price

                        # Color based on change
                        delta_color = "normal" if change >= 0 else "inverse"

                        st.metric(
                            label=f"{crypto}/USDT",
                            value=f"${price:,.2f}" if price > 1 else f"${price:.6f}",
                            delta=f"{change:.2f}%",
                            delta_color=delta_color
                        )

                        market_data.append({
                            'Symbol': crypto,
                            'Price': price,
                            'Change %': change,
                            'Volume': volume,
                            'High 24h': float(binance_data.get('highPrice', 0)),
                            'Low 24h': float(binance_data.get('lowPrice', 0))
                        })

            # Market data table with all cryptocurrencies
            st.subheader("Complete Market Data")
            if market_data:
                df = pd.DataFrame(market_data)
                df['Price'] = df['Price'].apply(lambda x: f"${x:,.2f}" if x > 1 else f"${x:.6f}")
                df['Change %'] = df['Change %'].apply(lambda x: f"{x:.2f}%")
                df['Volume'] = df['Volume'].apply(lambda x: f"${x:,.0f}")
                df['High 24h'] = df['High 24h'].apply(lambda x: f"${x:,.2f}" if x > 1 else f"${x:.6f}")
                df['Low 24h'] = df['Low 24h'].apply(lambda x: f"${x:,.2f}" if x > 1 else f"${x:.6f}")

                st.dataframe(df, use_container_width=True, height=400)

        with tab2:
            st.subheader("Advanced Price Charts")

            # Chart controls
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                selected_crypto = st.selectbox("Select cryptocurrency", CRYPTO_LIST)

            with col2:
                time_period = st.selectbox("Time period", [
                    "1d", "2d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y"
                ])

            with col3:
                interval_options = {
                    "1 minute": "1m",
                    "5 minutes": "5m",
                    "15 minutes": "15m",
                    "30 minutes": "30m",
                    "1 hour": "1h",
                    "4 hours": "4h",
                    "1 day": "1d"
                }
                selected_interval = st.selectbox("Interval", list(interval_options.keys()))
                interval = interval_options[selected_interval]

            with col4:
                chart_type = st.selectbox("Chart type", ["candlestick", "line", "area"])

            # Fetch and display chart data
            if st.button("Update Chart") or selected_crypto:
                with st.spinner(f"Loading {selected_crypto} chart data..."):
                    # Try Binance first for minute data
                    if interval in ["1m", "5m", "15m", "30m", "1h", "4h"]:
                        limit_map = {"1d": 1440, "2d": 2880, "5d": 7200, "7d": 10080}
                        limit = limit_map.get(time_period, 1000)
                        data = CryptoDataAPI.get_binance_klines(selected_crypto, interval, limit)
                    else:
                        data = CryptoDataAPI.get_yfinance_data(selected_crypto, time_period, interval)

                    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                        chart = create_price_chart(
                            data,
                            f"{selected_crypto} Price Chart ({selected_interval})",
                            chart_type
                        )
                        st.plotly_chart(chart, use_container_width=True)

                        # Display current stats
                        if len(data) > 0:
                            current_price = data['Close'].iloc[-1]
                            high_24h = data['High'].max()
                            low_24h = data['Low'].min()
                            volume_24h = data['Volume'].sum()

                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                            with stat_col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with stat_col2:
                                st.metric("24h High", f"${high_24h:.2f}")
                            with stat_col3:
                                st.metric("24h Low", f"${low_24h:.2f}")
                            with stat_col4:
                                st.metric("Volume", f"{volume_24h:,.0f}")
                    else:
                        st.error("Could not load chart data. Please try again.")

        with tab3:
            st.subheader("Market Screener")

            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                min_price = st.number_input("Min Price ($)", min_value=0.0, value=0.0)
            with col2:
                max_price = st.number_input("Max Price ($)", min_value=0.0, value=100000.0)
            with col3:
                min_change = st.number_input("Min 24h Change (%)", value=-100.0)

            # Apply filters and show results
            filtered_data = []
            for crypto in CRYPTO_LIST[:30]:  # Screen top 30
                binance_data = CryptoDataAPI.get_binance_data(crypto)
                if binance_data:
                    price = float(binance_data.get('lastPrice', 0))
                    change = float(binance_data.get('priceChangePercent', 0))

                    if min_price <= price <= max_price and change >= min_change:
                        filtered_data.append({
                            'Symbol': crypto,
                            'Price': f"${price:.2f}" if price > 1 else f"${price:.6f}",
                            'Change %': f"{change:.2f}%",
                            'Volume': f"${float(binance_data.get('volume', 0)):,.0f}"
                        })

            if filtered_data:
                st.dataframe(pd.DataFrame(filtered_data), use_container_width=True)
            else:
                st.info("No cryptocurrencies match the current filters.")

    elif page == "Buy Crypto":
        st.header("💰 Buy Cryptocurrency")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Purchase Options")

            crypto_to_buy = st.selectbox("Select cryptocurrency", CRYPTO_LIST)

            # Get current price
            binance_data = CryptoDataAPI.get_binance_data(crypto_to_buy)
            current_price = float(binance_data.get('lastPrice', 0)) if binance_data else 0

            if current_price > 0:
                st.info(f"Current {crypto_to_buy} price: ${current_price:,.2f}")

                purchase_type = st.radio("Purchase type", ["Market Order", "Limit Order"])

                if purchase_type == "Market Order":
                    amount_usd = st.number_input("Amount (USD)", min_value=1.0, max_value=st.session_state.cash_balance)
                    amount_crypto = amount_usd / current_price
                    st.info(f"You will receive: {amount_crypto:.6f} {crypto_to_buy}")

                    if st.button("Buy Now"):
                        success, message = simulate_trading("BUY", crypto_to_buy, amount_crypto, current_price)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

                else:  # Limit Order
                    limit_price = st.number_input("Limit price (USD)", min_value=0.01, value=current_price)
                    amount_crypto = st.number_input(f"Amount ({crypto_to_buy})", min_value=0.000001)

                    st.info(f"Total cost: ${limit_price * amount_crypto:.2f}")

                    if st.button("Place Limit Order"):
                        st.info("Limit order placed (simulated)")

        with col2:
            st.subheader("Account Balance")
            st.metric("Cash Balance", f"${st.session_state.cash_balance:,.2f}")

            st.subheader("Quick Buy")
            quick_amounts = [100, 500, 1000, 2500]
            for amount in quick_amounts:
                if st.button(f"Buy ${amount}"):
                    if amount <= st.session_state.cash_balance:
                        crypto_amount = amount / current_price
                        success, message = simulate_trading("BUY", crypto_to_buy, crypto_amount, current_price)
                        if success:
                            st.success(message)
                    else:
                        st.error("Insufficient funds")

    elif page == "Trading":
        st.header("📈 Advanced Trading")

        tab1, tab2, tab3 = st.tabs(["Spot Trading", "Order Book", "Trading History"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Buy Orders")
                buy_crypto = st.selectbox("Select crypto to buy", CRYPTO_LIST)
                binance_data = CryptoDataAPI.get_binance_data(buy_crypto)
                buy_price = float(binance_data.get('lastPrice', 0)) if binance_data else 0

                buy_amount = st.number_input("Amount to buy", min_value=0.000001, step=0.000001)
                buy_total = buy_amount * buy_price

                st.info(f"Total cost: ${buy_total:.2f}")

                if st.button("Execute Buy"):
                    success, message = simulate_trading("BUY", buy_crypto, buy_amount, buy_price)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

            with col2:
                st.subheader("Sell Orders")
                sell_crypto = st.selectbox("Select crypto to sell", CRYPTO_LIST)
                binance_data = CryptoDataAPI.get_binance_data(sell_crypto)
                sell_price = float(binance_data.get('lastPrice', 0)) if binance_data else 0

                max_sell = st.session_state.portfolio[sell_crypto]
                sell_amount = st.number_input(
                    f"Amount to sell (Max: {max_sell:.6f})",
                    min_value=0.0,
                    max_value=max_sell,
                    step=0.000001
                )
                sell_total = sell_amount * sell_price

                st.info(f"Total revenue: ${sell_total:.2f}")

                if st.button("Execute Sell"):
                    success, message = simulate_trading("SELL", sell_crypto, sell_amount, sell_price)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        with tab2:
            st.subheader("Order Book (Simulated)")
            selected_pair = st.selectbox("Trading Pair", [f"{crypto}/USDT" for crypto in CRYPTO_LIST[:20]])

            # Simulate order book data
            bid_data = []
            ask_data = []
            base_price = 50000  # Example price

            for i in range(10):
                bid_price = base_price - (i * 10)
                ask_price = base_price + (i * 10)
                bid_data.append([bid_price, np.random.uniform(0.1, 5.0)])
                ask_data.append([ask_price, np.random.uniform(0.1, 5.0)])

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Bids**")
                bid_df = pd.DataFrame(bid_data, columns=['Price', 'Amount'])
                st.dataframe(bid_df, use_container_width=True)

            with col2:
                st.write("**Asks**")
                ask_df = pd.DataFrame(ask_data, columns=['Price', 'Amount'])
                st.dataframe(ask_df, use_container_width=True)

        with tab3:
            st.subheader("Trading History")
            if st.session_state.trading_history:
                history_df = pd.DataFrame(st.session_state.trading_history)
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("No trading history yet")

    elif page == "Futures":
        st.header("⚡ Futures Trading")

        st.warning("⚠️ Futures trading involves high risk. This is a simulation.")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Open Position")

            futures_symbol = st.selectbox("Select futures contract", [f"{crypto}USDT" for crypto in CRYPTO_LIST[:20]])
            position_side = st.selectbox("Position", ["Long", "Short"])
            leverage = st.slider("Leverage", 1, 100, 10)
            position_size = st.number_input("Position size (USDT)", min_value=10.0)

            # Get current price
            symbol = futures_symbol.replace("USDT", "")
            binance_data = CryptoDataAPI.get_binance_data(symbol)
            entry_price = float(binance_data.get('lastPrice', 0)) if binance_data else 0

            if entry_price > 0:
                st.info(f"Entry price: ${entry_price:,.2f}")
                margin_required = position_size / leverage
                st.info(f"Margin required: ${margin_required:.2f}")

                if st.button("Open Position"):
                    if margin_required <= st.session_state.cash_balance:
                        st.session_state.futures_positions.append({
                            'symbol': futures_symbol,
                            'side': position_side,
                            'size': position_size,
                            'leverage': leverage,
                            'entry_price': entry_price,
                            'margin': margin_required,
                            'timestamp': datetime.now()
                        })
                        st.session_state.cash_balance -= margin_required
                        st.success(f"Opened {position_side} position for {futures_symbol}")
                    else:
                        st.error("Insufficient margin")

        with col2:
            st.subheader("Active Positions")
            if st.session_state.futures_positions:
                for i, position in enumerate(st.session_state.futures_positions):
                    with st.container():
                        st.write(f"**{position['symbol']} {position['side']}**")
                        st.write(f"Size: ${position['size']:.2f}")
                        st.write(f"Leverage: {position['leverage']}x")
                        st.write(f"Entry: ${position['entry_price']:.2f}")

                        if st.button(f"Close Position {i+1}"):
                            # Return margin (simplified)
                            st.session_state.cash_balance += position['margin']
                            st.session_state.futures_positions.pop(i)
                            st.success("Position closed")
                            st.rerun()
                        st.divider()
            else:
                st.info("No active positions")

    elif page == "Earn":
        st.header("💎 Earn Crypto")

        tab1, tab2, tab3 = st.tabs(["Staking", "Lending", "Liquidity Mining"])

        with tab1:
            st.subheader("Crypto Staking")

            staking_options = {
                'BTC': {'apy': 4.5, 'min_amount': 0.001},
                'ETH': {'apy': 5.2, 'min_amount': 0.01},
                'BNB': {'apy': 6.8, 'min_amount': 0.1},
                'SOL': {'apy': 7.5, 'min_amount': 1.0},
                'ADA': {'apy': 6.2, 'min_amount': 10.0},
                'DOT': {'apy': 8.1, 'min_amount': 1.0},
                'ATOM': {'apy': 9.5, 'min_amount': 1.0},
                'MATIC': {'apy': 7.8, 'min_amount': 100.0},
                'AVAX': {'apy': 6.9, 'min_amount': 0.5},
                'ALGO': {'apy': 5.5, 'min_amount': 10.0}
            }

            col1, col2 = st.columns([2, 1])

            with col1:
                for crypto, details in staking_options.items():
                    with st.container():
                        st.write(f"**{crypto} Staking**")
                        st.write(f"APY: {details['apy']:.1f}%")
                        st.write(f"Minimum: {details['min_amount']} {crypto}")

                        available = st.session_state.portfolio[crypto]
                        stake_amount = st.number_input(
                            f"Amount to stake ({crypto})",
                            min_value=0.0,
                            max_value=available,
                            key=f"stake_{crypto}"
                        )

                        if st.button(f"Stake {crypto}", key=f"btn_stake_{crypto}"):
                            if stake_amount >= details['min_amount'] and stake_amount <= available:
                                st.session_state.portfolio[crypto] -= stake_amount
                                st.session_state.staking_positions.append({
                                    'crypto': crypto,
                                    'amount': stake_amount,
                                    'apy': details['apy'],
                                    'start_date': datetime.now()
                                })
                                st.success(f"Staked {stake_amount} {crypto}")
                            else:
                                st.error("Invalid amount")
                        st.divider()

            with col2:
                st.subheader("Active Stakes")
                if st.session_state.staking_positions:
                    for i, stake in enumerate(st.session_state.staking_positions):
                        days_staked = (datetime.now() - stake['start_date']).days
                        rewards = stake['amount'] * (stake['apy'] / 100) * (days_staked / 365)

                        st.write(f"**{stake['crypto']}**")
                        st.write(f"Amount: {stake['amount']:.6f}")
                        st.write(f"Days: {days_staked}")
                        st.write(f"Rewards: {rewards:.6f}")

                        if st.button(f"Unstake {i+1}"):
                            st.session_state.portfolio[stake['crypto']] += stake['amount'] + rewards
                            st.session_state.staking_positions.pop(i)
                            st.success(f"Unstaked with rewards!")
                            st.rerun()
                        st.divider()
                else:
                    st.info("No active staking positions")

        with tab2:
            st.subheader("Crypto Lending")
            st.info("🚧 Feature coming soon - Lend your crypto to earn interest")

            lending_rates = {
                'BTC': 3.2,
                'ETH': 4.1,
                'BNB': 5.5,
                'USDT': 8.2
            }

            for crypto, rate in lending_rates.items():
                st.write(f"**{crypto}**: {rate:.1f}% APY")

        with tab3:
            st.subheader("Liquidity Mining")
            st.info("🚧 Feature coming soon - Provide liquidity to earn fees")

            pools = [
                {'pair': 'BTC/ETH', 'apy': 12.5, 'tvl': '$2.5M'},
                {'pair': 'ETH/BNB', 'apy': 15.2, 'tvl': '$1.8M'},
                {'pair': 'BNB/SOL', 'apy': 18.7, 'tvl': '$950K'}
            ]

            pool_df = pd.DataFrame(pools)
            st.dataframe(pool_df, use_container_width=True)

    elif page == "AI Analysis":
        st.header("🤖 AI Market Analysis")

        if not gemini_api_key:
            st.warning("Please enter your Gemini API key in the sidebar to use AI features")
        else:
            tab1, tab2, tab3 = st.tabs(["Market Analysis", "Trading Advice", "Risk Assessment"])

            with tab1:
                st.subheader("AI Market Insights")

                if st.button("Generate Market Analysis"):
                    with st.spinner("Analyzing market data..."):
                        # Gather market data
                        market_data = {}
                        for crypto in ['BTC', 'ETH', 'BNB', 'SOL']:
                            binance_data = CryptoDataAPI.get_binance_data(crypto)
                            if binance_data:
                                market_data[crypto] = {
                                    'price': binance_data.get('lastPrice'),
                                    'change_24h': binance_data.get('priceChangePercent'),
                                    'volume': binance_data.get('volume')
                                }

                        analysis = ai_agent.analyze_market(market_data)
                        st.write(analysis)

            with tab2:
                st.subheader("Personalized Trading Advice")

                advice_crypto = st.selectbox("Get advice for:", CRYPTO_LIST[:20])

                if st.button("Get AI Trading Advice"):
                    with st.spinner("Generating advice..."):
                        binance_data = CryptoDataAPI.get_binance_data(advice_crypto)
                        if binance_data:
                            current_price = float(binance_data.get('lastPrice', 0))
                            advice = ai_agent.get_trading_advice(
                                advice_crypto,
                                current_price,
                                st.session_state.portfolio
                            )
                            st.write(advice)

            with tab3:
                st.subheader("Portfolio Risk Assessment")
                st.info("🚧 AI Risk Analysis feature coming soon")

                # Portfolio overview
                portfolio_value = 0
                for crypto, amount in st.session_state.portfolio.items():
                    if amount > 0:
                        binance_data = CryptoDataAPI.get_binance_data(crypto)
                        if binance_data:
                            price = float(binance_data.get('lastPrice', 0))
                            value = amount * price
                            portfolio_value += value
                            st.write(f"{crypto}: {amount:.6f} (${value:.2f})")

                total_value = portfolio_value + st.session_state.cash_balance
                st.metric("Total Portfolio Value", f"${total_value:.2f}")

    elif page == "Portfolio":
        st.header("💼 Portfolio Overview")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Holdings")

            portfolio_data = []
            total_value = 0

            for crypto, amount in st.session_state.portfolio.items():
                if amount > 0:
                    binance_data = CryptoDataAPI.get_binance_data(crypto)
                    if binance_data:
                        price = float(binance_data.get('lastPrice', 0))
                        value = amount * price
                        change_24h = float(binance_data.get('priceChangePercent', 0))
                        total_value += value

                        portfolio_data.append({
                            'Crypto': crypto,
                            'Amount': f"{amount:.6f}",
                            'Price': f"${price:,.2f}",
                            'Value': f"${value:.2f}",
                            '24h Change': f"{change_24h:.2f}%"
                        })

            if portfolio_data:
                st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True)

                # Portfolio pie chart
                if len(portfolio_data) > 1:
                    values = [float(item['Value'].replace('$', '').replace(',', '')) for item in portfolio_data]
                    labels = [item['Crypto'] for item in portfolio_data]

                    fig = px.pie(values=values, names=labels, title="Portfolio Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No crypto holdings yet")

        with col2:
            st.subheader("Account Summary")

            st.metric("Cash Balance", f"${st.session_state.cash_balance:,.2f}")
            st.metric("Crypto Value", f"${total_value:.2f}")
            st.metric("Total Portfolio", f"${st.session_state.cash_balance + total_value:,.2f}")

            # Performance metrics
            st.subheader("Performance")
            initial_balance = 10000.0
            current_total = st.session_state.cash_balance + total_value
            pnl = current_total - initial_balance
            pnl_pct = (pnl / initial_balance) * 100

            st.metric("P&L", f"${pnl:.2f}", f"{pnl_pct:.2f}%")

            # Quick actions
            st.subheader("Quick Actions")
            if st.button("Reset Portfolio"):
                st.session_state.portfolio = {crypto: 0.0 for crypto in CRYPTO_LIST}
                st.session_state.cash_balance = 10000.0
                st.session_state.trading_history = []
                st.success("Portfolio reset!")

if __name__ == "__main__":

    main()
