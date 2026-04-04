# P2-ETF-REGIMEFLOW

## 🧠 RegimeFlow ETF Engine

**Regime-Conditioned Cross-Sectional Rotation for ETFs**

---

## 🚀 Overview

**RegimeFlow** is a systematic ETF allocation engine that selects the highest expected-return ETF **conditional on the current macro regime**.

Unlike traditional models that attempt to predict returns directly, RegimeFlow operates on a more robust principle:

> Market behavior is **regime-dependent**, and ETF performance varies systematically across regimes.

---

## 🎯 Core Idea

```text
Detect current macro regime → 
Estimate conditional return distributions → 
Select ETF with highest expected payoff
```

---

## 📊 Data Used

Dataset:
`P2SAMAPA/fi-etf-macro-signal-master-data`

### 🟦 ETF Universe

#### Fixed Income / Commodities

* TLT (Treasuries)
* LQD (Investment Grade Credit)
* HYG (High Yield Credit)
* VNQ (Real Estate)
* GLD (Gold)
* SLV (Silver)

Benchmark: AGG *(not traded)*

---

#### Equity

* SPY (S&P 500 — benchmark only)
* QQQ (NASDAQ 100)
* XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU
* GDX (Gold Miners)
* XME (Metals & Mining)
* IWM (Russell 2000)

Benchmark: SPY *(not traded)*

---

### 🟨 Macro Features (Regime Drivers)

* VIX → volatility regime
* DXY → dollar strength
* T10Y2Y → yield curve
* TBILL_3M → risk-free rate
* IG_SPREAD → credit conditions
* HY_SPREAD → risk stress

---

## 🧱 Engine Architecture

```text
Data Loader
    ↓
Regime Detection (KMeans Clustering)
    ↓
Regime Labeling (per day)
    ↓
Conditional Return Distributions
    ↓
ETF Scoring (μ × P(positive))
    ↓
Portfolio Decision Logic
    ↓
Backtest + Output
```

---

## 🧠 Regime Detection

* Uses **KMeans clustering (K=4)** on macro features
* Each day is assigned a **regime ID**
* Regimes implicitly capture:

  * Risk-on / risk-off
  * Inflation / deflation
  * Stress / recovery phases

---

## 📈 Conditional Modeling

For each regime:

```text
P(Return | ETF, Regime)
```

Built using **empirical distributions**:

* Historical returns filtered by regime
* Bootstrap sampling (100 samples per ETF)

---

## 🎯 ETF Scoring

For each ETF:

```text
Score = Mean(Return) × Probability(Return > 0)
```

This balances:

* Expected return
* Directional confidence

---

## 🏆 Selection Logic

* Rank ETFs by score
* Select highest scoring ETF
* Track full cross-sectional distribution

---

## 🛡️ Portfolio Rules

### 🔹 Transaction Cost

* 12 bps penalty applied when switching ETFs

---

### 🔹 Trailing Stop Loss (TSL)

* Trigger: **2-day cumulative return < -12%**
* Action: move to **CASH**

---

### 🔹 CASH Mode

* Earns **3M T-Bill rate**
* Exit condition (future upgrade): Z-score normalization

---

## 🔁 Backtest Engine

* Daily walk-forward simulation
* Uses regime-conditioned scoring
* Outputs:

  * Equity curve
  * Strategy performance

---

## 📦 Outputs

Each run generates a JSON file:

```text
outputs/regimeflow_YYYY-MM-DD.json
```

And uploads to:

`P2SAMAPA/p2-etf-regimeflow-results`

---

### Output Structure

```json
{
  "date": "...",
  "next_trading_day": "...",
  "FI": {
    "pick": "...",
    "scores": {...}
  },
  "EQ": {
    "pick": "...",
    "scores": {...}
  },
  "samples_fi": {...},
  "equity_curve": [...]
}
```

---

## 📊 Streamlit UI

The app provides:

* 📌 Daily ETF picks (FI + Equity)
* 📊 Score tables
* 📉 Return distribution charts
* 📈 Equity curve
* 🗂 Signal history

---

## ⚙️ Deployment

### GitHub Actions

* Runs daily (weekday schedule)
* Executes `run.py`
* Pushes results to Hugging Face dataset

---

### Streamlit

* Pulls latest JSON output
* Displays signals and analytics

---

## 🧠 Why This Works

Traditional models assume:

```text
Returns are stationary ❌
```

RegimeFlow assumes:

```text
Return distributions depend on macro regime ✅
```

This allows:

* Better adaptation to market shifts
* More stable cross-sectional selection
* Reduced overfitting

---

## ⚠️ Limitations

* Regime boundaries are approximate
* Clustering is unsupervised
* Performance depends on macro feature quality
* No explicit volatility targeting (yet)

---

## 🔥 Future Enhancements

* Regime transition modeling (Markov chains)
* Multi-window ensemble (like DIFFMAP)
* Regime stability filters
* Cross-asset hedging
* Meta-model stacking

---

## ⚠️ Disclaimer

This project is for research and educational purposes only.
It does **not** constitute financial advice.

---

## 👤 Author

P2SAMAPA
