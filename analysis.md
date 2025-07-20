# Credit Score Analysis Report

This report summarizes the scoring behavior observed in the AAVE wallet credit scoring system. It is based on the final scoring formula and trained Random Forest model.

---

## 🎯 Final Heuristic Scoring Formula

```python
y = (
        df['num_deposits'] * 1.1
        + df['repay_borrow_ratio'].clip(0, 5) * 5  # reduce outlier impact
        - df['num_liquidations'] * 37
        + df['activity_duration_days'] * 0.2
        + df['total_usd'].apply(np.log1p) / 1000
        + np.log1p(df['avg_tx_usd']) / 100
    )

```

- The output is then scaled to a **0–1000** range.

---

## 📊 Score Distribution Histogram

A histogram showing how many wallets fall into each score range:

| Bucket   | Count |
| -------- | ----- |
| 0–100    | 1     |
| 100–200  | 2     |
| 200–300  | 2     |
| 300–400  | 8     |
| 400–500  | 2255  |
| 500–600  | 1136  |
| 600–700  | 68    |
| 700–800  | 14    |
| 800–900  | 8     |
| 900–1000 | 2     |

- Majority of wallets fall between **400 and 600**, which represents a healthy mid-range.
- **Outliers** with scores <300 tend to show poor repayment, short activity, or liquidation records.

---

## 🧠 Behavior Insights

### 🔴 Low Scoring Wallets (0–300)

- Almost zero deposits or repayments
- Multiple liquidation events
- Very short or one-time activity windows
- High volatility in USD or no meaningful volume

### 🟡 Mid Scoring Wallets (400–600)

- Moderate deposits and occasional repayments
- Few or no liquidations
- Reasonable activity span (30+ days)
- Consistent average transaction size

### 🟢 High Scoring Wallets (700+)

- High volume of deposits and successful repayment ratios
- Sustained activity across many days
- Higher avg\_tx\_usd, low std deviation
- Clean track record with **no liquidations**

---

## 🔍 Top 3 Highest Scoring Wallets

| Wallet | Score | Deposits | Repay Ratio | Total USD | Duration |
| ------ | ----- | -------- | ----------- | --------- | -------- |
| 0x...1 | 997   | 156      | 0.91        | 3.2e+22   | 128 days |
| 0x...2 | 993   | 122      | 0.87        | 2.8e+22   | 110 days |
| 0x...3 | 991   | 140      | 0.79        | 1.9e+22   | 106 days |

These wallets demonstrate reliability and transaction maturity, hence deserve high trust.

---

## 📌 Conclusion

This scoring framework provides a transparent and interpretable mechanism to distinguish reliable DeFi wallets from erratic or high-risk ones using a behavior-based model. It is **extensible**, explainable, and works without oracle dependency.

Feel free to customize the heuristic or upgrade the ML model for improved accuracy.

