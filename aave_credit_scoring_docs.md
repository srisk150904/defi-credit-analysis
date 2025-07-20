# Repository: AAVE Wallet Credit Scoring

This repository implements an end-to-end pipeline to estimate **credit scores** for DeFi wallets interacting with the **AAVE v2 protocol**. The model uses on-chain transaction behavior to assign scores from **0 to 1000**, reflecting each wallet's relative financial reliability.

---

## 🔧 Tech Stack

- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Matplotlib, scikit-learn, LightGBM (optional), Joblib

---

## 📂 Repository Structure

```
AAVE-Credit-Scoring/
├── main.py                  # Entry point for end-to-end flow
├── parse_data.py           # JSON parsing and conversion to DataFrame
├── feature_engineering.py  # Wallet-level aggregation logic
├── model.py                # ML model definition and scoring
├── evaluate.py             # Visuals, summaries, and wallet inspection
├── utils.py                # Utility functions like scaling
├── data/                   # Input JSON and processed CSVs
├── outputs/                # Model files, plots, etc.
├── README.md               # Project overview
└── analysis.md             # Wallet score analysis and insights
```

---

## 📌 Methodology Summary

### ✅ 1. Data Parsing

- Raw AAVE transaction data in JSON format is loaded
- Converted into a flat `pandas.DataFrame` using `parse_data.py`

### ✅ 2. Feature Engineering

- Wallet-level features include:
  - Transaction counts (`num_deposits`, `num_borrows`, etc.)
  - Total and average USD moved
  - Repayment behavior
  - Duration and frequency of activity
  - USD volatility (`stddev`, `avg_tx_usd`)

### ✅ 3. Heuristic Label Creation

A pseudo-label is generated based on:

```python
    y = (
        df['num_deposits'] * 1.2
        + df['repay_borrow_ratio'] * 7
        - df['num_liquidations'] * 30
        + df['activity_duration_days'] * 0.3
        + np.log1p(df['total_usd']) / 1000
        + np.log1p(df['avg_tx_usd']) / 100
    )
```

- Scaled to `[0, 1000]` using MinMax normalization

### ✅ 4. Model Training

- Base model: `RandomForestRegressor`
- Optional: Replaceable with `LightGBMRegressor`
- Trained using `train_model()` in `model.py`

### ✅ 5. Scoring

- New wallets are scored using the trained model
- Scores are saved with wallet metadata for evaluation

### ✅ 6. Evaluation

- Score distributions are plotted
- Wallets are grouped by score buckets (0–100, ..., 900–1000)
- Behavior of wallets across score ranges is analyzed

---

## 🏁 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Optional: Evaluate and analyze output
python -m evaluate
```

---

## 📈 Credit Scoring Output Example

| wallet\_address | credit\_score |
| --------------- | ------------- |
| 0xabc123...     | 685           |
| 0xdef456...     | 204           |
| ...             | ...           |

---

## 📊 Result Summary

- Wallets with higher scores showed:
  - High deposits and repayment activity
  - Long consistent durations
  - Low volatility and liquidations
- Lower scores were assigned to erratic, low-activity, or liquidated wallets

For more detailed behavior analysis, see [analysis.md](analysis.md).

---

## 📃 License

MIT License

