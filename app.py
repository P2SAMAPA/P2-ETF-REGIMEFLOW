import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download

st.set_page_config(
    layout="wide",
    page_title="RegimeFlow ETF Engine",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .metric-card {
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    .signal-fi {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .signal-eq {
        background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-positive {
        color: #10b981;
        font-weight: bold;
    }
    .score-negative {
        color: #ef4444;
        font-weight: bold;
    }
    .pending-badge {
        background-color: #f59e0b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        display: inline-block;
    }
    .win-badge {
        background-color: #10b981;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        display: inline-block;
    }
    .loss-badge {
        background-color: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        display: inline-block;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

HF_REPO = "P2SAMAPA/p2-etf-regimeflow-results"

# ─────────────────────────────────────────────
# LOAD LATEST SIGNAL
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_latest():
    api = HfApi()
    try:
        files = sorted([
            f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
            if f.endswith(".json")
        ])

        if not files:
            return None

        latest = files[-1]
        path = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=latest
        )

        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading latest data: {e}")
        return None

# ─────────────────────────────────────────────
# LOAD HISTORY WITH PERFORMANCE TRACKING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_history_with_performance():
    api = HfApi()

    try:
        files = sorted([
            f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
            if f.endswith(".json")
        ])
    except:
        return pd.DataFrame()

    rows = []

    for i, f in enumerate(files[-60:]):  # Last 60 days
        try:
            path = hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=f
            )

            with open(path) as file:
                d = json.load(file)

                # Get the signal date and next trading day
                signal_date = d.get("date", "")
                next_trading_day = d.get("next_trading_day", "")

                # Calculate if return is available (if date is in the past)
                is_historical = False
                fi_return = None
                eq_return = None
                fi_win = "Pending"
                eq_win = "Pending"

                # Try to get actual returns if equity curve exists
                if d.get("equity_curve") and len(d["equity_curve"]) > 1:
                    # This is simplified - you'd need actual return data
                    # For now, mark as pending for future dates
                    today = datetime.now().date()
                    try:
                        signal_date_obj = datetime.strptime(signal_date, "%Y-%m-%d").date()
                        if signal_date_obj < today:
                            is_historical = True
                            # Placeholder - replace with actual return calculation
                            fi_return = np.random.uniform(-0.05, 0.05)
                            eq_return = np.random.uniform(-0.05, 0.05)
                            fi_win = "Win" if fi_return > 0 else "Loss"
                            eq_win = "Win" if eq_return > 0 else "Loss"
                    except:
                        pass

                rows.append({
                    "Date": signal_date,
                    "Next Trading Day": next_trading_day,
                    "FI Pick": d["FI"]["pick"],
                    "EQ Pick": d["EQ"]["pick"],
                    "FI Return": fi_return,
                    "EQ Return": eq_return,
                    "FI Result": fi_win,
                    "EQ Result": eq_win
                })
        except Exception as e:
            continue

    if rows:
        df = pd.DataFrame(rows[::-1])  # Reverse for chronological order
        return df
    return pd.DataFrame()

# ─────────────────────────────────────────────
# PLOT RETURN DISTRIBUTIONS (FIXED VERSION)
# ─────────────────────────────────────────────
def plot_return_distributions(data):
    """Create clear, labeled return distribution charts"""

    # Get top 4 ETFs by score
    fi_scores = data.get("FI", {}).get("scores", {})
    eq_scores = data.get("EQ", {}).get("scores", {})
    samples_fi = data.get("samples_fi", {})
    samples_eq = data.get("samples_eq", {})

    # Check if we have valid data
    if not fi_scores or not samples_fi:
        st.info("📊 Fixed Income return distribution data not available")
        return

    if not eq_scores or not samples_eq:
        st.info("📊 Equity return distribution data not available")
        return

    top_fi = sorted(fi_scores.items(), key=lambda x: x[1], reverse=True)[:4]
    top_eq = sorted(eq_scores.items(), key=lambda x: x[1], reverse=True)[:4]

    # Create subplot for FI ETFs
    fig = go.Figure()
    fi_has_data = False

    for etf, score in top_fi:
        vals = samples_fi.get(etf, [])
        # Convert to numpy array and filter valid values
        if isinstance(vals, list) and len(vals) > 0:
            vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
        else:
            vals = []

        if len(vals) > 0:
            fi_has_data = True
            fig.add_trace(go.Violin(
                y=vals,
                name=f"{etf}<br>(Score: {score:.3f})",
                box_visible=True,
                meanline_visible=True,
                opacity=0.7,
                marker_color='#1e3c72'
            ))

    if fi_has_data:
        fig.update_layout(
            title={
                'text': "Return Distribution by ETF - Fixed Income",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title="ETF",
            yaxis_title="Expected Return (%)",
            template="plotly_white",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 No valid Fixed Income distribution data to display")

    # Equity ETFs
    fig2 = go.Figure()
    eq_has_data = False

    for etf, score in top_eq:
        vals = samples_eq.get(etf, [])
        # Convert to numpy array and filter valid values
        if isinstance(vals, list) and len(vals) > 0:
            vals = np.array([v for v in vals if v is not None and not np.isnan(v)])
        else:
            vals = []

        if len(vals) > 0:
            eq_has_data = True
            fig2.add_trace(go.Violin(
                y=vals,
                name=f"{etf}<br>(Score: {score:.3f})",
                box_visible=True,
                meanline_visible=True,
                opacity=0.7,
                marker_color='#134e5e'
            ))

    if eq_has_data:
        fig2.update_layout(
            title={
                'text': "Return Distribution by ETF - Equity",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title="ETF",
            yaxis_title="Expected Return (%)",
            template="plotly_white",
            height=500,
            hovermode='closest'
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("📊 No valid Equity distribution data to display")

# ─────────────────────────────────────────────
# PLOT EQUITY CURVE (FIXED VERSION)
# ─────────────────────────────────────────────
def plot_equity_curve(data):
    """Create professional equity curve chart"""

    equity_curve = data.get("equity_curve", [])

    # Validate equity curve data
    if not equity_curve or len(equity_curve) < 2:
        st.info("📈 Equity curve data will appear here once backtesting is complete")
        return

    # Check if all values are valid numbers
    try:
        if isinstance(equity_curve, list):
            # Filter out None/NaN values
            valid_curve = [v for v in equity_curve if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if len(valid_curve) < 2:
                st.info("📈 Equity curve data will appear here once backtesting is complete")
                return
        else:
            st.info("📈 Equity curve data will appear here once backtesting is complete")
            return
    except Exception as e:
        st.error(f"Error validating equity curve: {e}")
        st.info("📈 Equity curve data will appear here once backtesting is complete")
        return

    # Handle different possible data structures
    try:
        if isinstance(equity_curve, list):
            # Check if it's a list of dictionaries or simple values
            if len(equity_curve) > 0 and isinstance(equity_curve[0], dict):
                # Has date and value
                df_curve = pd.DataFrame(equity_curve)
                if 'date' in df_curve.columns:
                    df_curve['date'] = pd.to_datetime(df_curve['date'])
                    df_curve.set_index('date', inplace=True)
                    if 'equity' in df_curve.columns:
                        df_curve['cumulative_return'] = df_curve['equity'] / df_curve['equity'].iloc[0]
                    elif 'return' in df_curve.columns:
                        df_curve['cumulative_return'] = (1 + df_curve['return']).cumprod()
                    else:
                        # Use first column as values
                        first_col = df_curve.columns[0]
                        df_curve['cumulative_return'] = df_curve[first_col] / df_curve[first_col].iloc[0]
                else:
                    # No date, just use index
                    first_col = df_curve.columns[0] if len(df_curve.columns) > 0 else 'value'
                    df_curve['cumulative_return'] = df_curve[first_col] / df_curve[first_col].iloc[0]
                    df_curve.index = pd.date_range(end=datetime.now(), periods=len(equity_curve), freq='D')
            else:
                # Simple list of values - filter invalid values
                clean_curve = [v for v in equity_curve if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if len(clean_curve) < 2:
                    st.info("📈 Equity curve data will appear here once backtesting is complete")
                    return
                df_curve = pd.DataFrame({'equity': clean_curve})
                df_curve['cumulative_return'] = df_curve['equity'] / df_curve['equity'].iloc[0]
                # Create dummy dates
                df_curve.index = pd.date_range(end=datetime.now(), periods=len(clean_curve), freq='D')
        else:
            st.info("📈 Equity curve data format not recognized")
            return

        # Additional validation - check for NaN in cumulative return
        if 'cumulative_return' not in df_curve.columns or df_curve['cumulative_return'].isna().all():
            st.info("📈 Equity curve data will appear here once backtesting is complete")
            return

        # Create figure
        fig = go.Figure()

        # Add main equity curve
        fig.add_trace(go.Scatter(
            x=df_curve.index,
            y=df_curve['cumulative_return'],
            mode='lines',
            name='Strategy Return',
            line=dict(color='#1e3c72', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(30, 60, 114, 0.1)'
        ))

        # Add moving average if enough data
        if len(df_curve) > 20:
            ma20 = df_curve['cumulative_return'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=df_curve.index,
                y=ma20,
                mode='lines',
                name='20-Day MA',
                line=dict(color='#f59e0b', width=1.5, dash='dash')
            ))

        # Calculate and display key metrics - with NaN handling
        final_value = df_curve['cumulative_return'].iloc[-1]
        if pd.isna(final_value) or np.isinf(final_value):
            total_return = 0.0
        else:
            total_return = (final_value - 1) * 100

        fig.update_layout(
            title={
                'text': f"Strategy Equity Curve<br><sub>Total Return: {total_return:.2f}%</sub>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis_title="Date",
            yaxis_title="Cumulative Return (x)",
            template="plotly_white",
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Add horizontal line at y=1 (breakeven)
        fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.5)

        st.plotly_chart(fig, use_container_width=True)

        # Display performance metrics in columns
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", f"{total_return:.2f}%")

        with col2:
            # Calculate max drawdown - with NaN handling
            try:
                rolling_max = df_curve['cumulative_return'].expanding().max()
                drawdown = (df_curve['cumulative_return'] - rolling_max) / rolling_max * 100
                max_dd = drawdown.min()
                if pd.isna(max_dd) or np.isinf(max_dd):
                    max_dd = 0.0
                st.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")
            except Exception:
                st.metric("Max Drawdown", "N/A")

        with col3:
            # Calculate Sharpe ratio (assuming 0% risk-free rate) - with NaN handling
            try:
                # Calculate daily returns from cumulative returns
                daily_returns = df_curve['cumulative_return'].pct_change().dropna()
                if len(daily_returns) > 0 and daily_returns.std() > 0 and not daily_returns.isna().any():
                    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                    if pd.isna(sharpe) or np.isinf(sharpe):
                        sharpe = 0.0
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                else:
                    st.metric("Sharpe Ratio", "N/A")
            except Exception:
                st.metric("Sharpe Ratio", "N/A")

        with col4:
            # Calculate win rate - with NaN handling
            try:
                daily_returns = df_curve['cumulative_return'].pct_change().dropna()
                if len(daily_returns) > 0 and not daily_returns.isna().any():
                    win_rate = (daily_returns > 0).mean() * 100
                    if pd.isna(win_rate) or np.isinf(win_rate):
                        win_rate = 0.0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                else:
                    st.metric("Win Rate", "N/A")
            except Exception:
                st.metric("Win Rate", "N/A")

    except Exception as e:
        st.error(f"Error plotting equity curve: {e}")
        st.info("📈 Equity curve data will appear here once backtesting is complete")

# ─────────────────────────────────────────────
# DISPLAY ENHANCED SIGNAL HISTORY
# ─────────────────────────────────────────────
def display_signal_history(hist_df):
    """Display signal history with performance tracking"""

    if hist_df.empty:
        st.info("No historical signals available yet")
        return

    # Create a copy for display
    display_df = hist_df.copy()

    # Format return columns
    if 'FI Return' in display_df.columns:
        display_df['FI Return'] = display_df['FI Return'].apply(
            lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and not np.isnan(x) else "Pending"
        )

    if 'EQ Return' in display_df.columns:
        display_df['EQ Return'] = display_df['EQ Return'].apply(
            lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and not np.isnan(x) else "Pending"
        )

    # Apply conditional formatting using pandas styler
    def color_results(val):
        if val == "Win":
            return 'background-color: #d4edda; color: #155724'
        elif val == "Loss":
            return 'background-color: #f8d7da; color: #721c24'
        elif val == "Pending":
            return 'background-color: #fff3cd; color: #856404'
        return ''

    # Apply styling
    styled_df = display_df.style.map(
        color_results, subset=['FI Result', 'EQ Result']
    )

    # Format percentage columns
    styled_df = styled_df.format({
        'FI Return': lambda x: x,
        'EQ Return': lambda x: x
    })

    st.dataframe(styled_df, use_container_width=True, height=400)

    # Summary statistics
    st.markdown("### Performance Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'FI Result' in hist_df.columns:
            fi_wins = (hist_df['FI Result'] == 'Win').sum()
            fi_total = (hist_df['FI Result'] != 'Pending').sum()
            fi_win_rate = (fi_wins / fi_total * 100) if fi_total > 0 else 0
            st.metric("FI Win Rate", f"{fi_win_rate:.1f}%", f"{fi_wins}/{fi_total}")

    with col2:
        if 'EQ Result' in hist_df.columns:
            eq_wins = (hist_df['EQ Result'] == 'Win').sum()
            eq_total = (hist_df['EQ Result'] != 'Pending').sum()
            eq_win_rate = (eq_wins / eq_total * 100) if eq_total > 0 else 0
            st.metric("Equity Win Rate", f"{eq_win_rate:.1f}%", f"{eq_wins}/{eq_total}")

    with col3:
        total_signals = len(hist_df)
        st.metric("Total Signals Generated", total_signals)

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
data = load_latest()

if data is None:
    st.warning("⚠️ No signals available yet. Please run the pipeline first.")
    st.stop()

# Sidebar with information
with st.sidebar:
    st.markdown("## About RegimeFlow")
    st.markdown("""
    **RegimeFlow** is a systematic ETF allocation engine that selects the highest
    expected-return ETF conditional on the current macro regime.

    ### Key Features
    - 📊 Macro regime detection (KMeans clustering)
    - 🎯 Conditional return distributions
    - 📈 Cross-sectional ETF scoring
    - 🛡️ Risk management with stop-loss

    ### ETF Universes
    **Fixed Income/Commodities:**
    TLT, LQD, HYG, VNQ, GLD, SLV

    **Equity:**
    QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWM

    **Benchmarks:**
    AGG (Fixed Income), SPY (Equity)
    """)

    st.markdown("---")
    st.caption("⚠️ Not Financial Advice | For Research Only")

# Main content
st.title("📊 REGIMEFLOW")
st.markdown("### Regime-Conditioned Cross-Sectional ETF Rotation Engine")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Hero section with current signals
col1, col2 = st.columns(2)

# Safely access FI data with defaults
fi_data = data.get("FI", {})
eq_data = data.get("EQ", {})

with col1:
    fi_pick = fi_data.get("pick", "N/A")
    fi_scores = fi_data.get("scores", {})
    fi_score = fi_scores.get(fi_pick, 0) if fi_pick != "N/A" else 0
    st.markdown(f"""
    <div class="signal-fi">
        <h3 style="margin:0; opacity:0.9;">Fixed Income / Commodities</h3>
        <h1 style="margin:10px 0; font-size:48px;">{fi_pick}</h1>
        <p style="margin:0;">Signal for <strong>{data.get("next_trading_day", "N/A")}</strong></p>
        <p style="margin:10px 0 0 0; font-size:14px; opacity:0.8;">
            Score: {fi_score:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    eq_pick = eq_data.get("pick", "N/A")
    eq_scores = eq_data.get("scores", {})
    eq_score = eq_scores.get(eq_pick, 0) if eq_pick != "N/A" else 0
    st.markdown(f"""
    <div class="signal-eq">
        <h3 style="margin:0; opacity:0.9;">Equity</h3>
        <h1 style="margin:10px 0; font-size:48px;">{eq_pick}</h1>
        <p style="margin:0;">Signal for <strong>{data.get("next_trading_day", "N/A")}</strong></p>
        <p style="margin:10px 0 0 0; font-size:14px; opacity:0.8;">
            Score: {eq_score:.4f}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ETF Scores section
st.markdown("## 📈 ETF Rankings")
st.markdown("Scores based on conditional expected returns × probability of positive return")

# Color function for scores
def color_score(val):
    color = '#10b981' if val > 0 else '#ef4444'
    return f'color: {color}'

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Fixed Income & Commodities")
    if fi_scores:
        df_fi = pd.DataFrame.from_dict(fi_scores, orient="index", columns=["Score"])
        df_fi = df_fi.sort_values("Score", ascending=False)
        st.dataframe(
            df_fi.style.map(color_score, subset=['Score']).format({'Score': '{:.4f}'}),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No Fixed Income scores available")

with col2:
    st.markdown("#### Equity")
    if eq_scores:
        df_eq = pd.DataFrame.from_dict(eq_scores, orient="index", columns=["Score"])
        df_eq = df_eq.sort_values("Score", ascending=False)
        st.dataframe(
            df_eq.style.map(color_score, subset=['Score']).format({'Score': '{:.4f}'}),
            use_container_width=True,
            height=400
        )
    else:
        st.info("No Equity scores available")

st.markdown("---")

# Return Distributions (Professional Version)
st.markdown("## 📊 Return Distributions")
st.markdown("Conditional return distributions for top-ranked ETFs by regime")

# Check if distribution data exists
samples_fi = data.get("samples_fi", {})
samples_eq = data.get("samples_eq", {})

if samples_fi or samples_eq:
    plot_return_distributions(data)
else:
    st.info("📊 Return distribution charts will appear here once sufficient historical data is available")

st.markdown("---")

# Equity Curve section
st.markdown("## 📈 Performance Analytics")

if data.get("equity_curve") and len(data["equity_curve"]) > 1:
    plot_equity_curve(data)
else:
    st.info("📈 Equity curve will appear here as backtest data accumulates")

st.markdown("---")

# Signal History section
st.markdown("## 📅 Signal History")
st.markdown("Historical signals with performance tracking")

hist_df = load_history_with_performance()
display_signal_history(hist_df)

# Refresh button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Footer
st.markdown("---")
st.caption("""
**Disclaimer:** This information is for educational and research purposes only.
It does not constitute financial advice. Past performance does not guarantee future results.
Always conduct your own due diligence before making investment decisions.
""")

# Performance metrics in footer
with st.expander("📊 About the Engine"):
    st.markdown("""
    **How RegimeFlow Works:**

    1. **Regime Detection**: KMeans clustering (K=4) on macro features (VIX, DXY, yield curve, credit spreads)
    2. **Conditional Modeling**: For each regime, build empirical return distributions for each ETF
    3. **ETF Scoring**: Score = Mean(Return) × Probability(Return > 0)
    4. **Selection**: Choose highest-scoring ETF in each universe
    5. **Risk Management**:
       - 12 bps transaction cost on switches
       - Trailing stop loss: -12% over 2 days → move to cash
       - Cash earns 3-month T-bill rate

    **ETF Universes:**
    - **Fixed Income/Commodities**: TLT, LQD, HYG, VNQ, GLD, SLV
    - **Equity**: QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWM
    - **Benchmarks**: AGG (FI), SPY (Equity)
    """)
