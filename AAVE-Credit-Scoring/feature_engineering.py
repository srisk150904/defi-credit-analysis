# src/feature_engineering.py

import pandas as pd
import numpy as np

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format raw transaction data.
    - Converts timestamp to datetime
    - Filters valid actions
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Keep only relevant actions
    allowed_actions = ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']
    df = df[df['action'].isin(allowed_actions)]

    return df


def aggregate_wallet_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups transaction data by wallet and computes features including:
    - counts
    - total & average USD value
    - volatility (stddev)
    - repay/borrow behavior
    - activity duration
    - tx frequency
    """

    df = df.sort_values(by=['wallet_address', 'timestamp'])

    features = []

    for wallet, group in df.groupby('wallet_address'):
        row = {}
        row['wallet_address'] = wallet
        row['tx_count'] = len(group)
        row['num_deposits'] = (group['action'] == 'deposit').sum()
        row['num_borrows'] = (group['action'] == 'borrow').sum()
        row['num_repays'] = (group['action'] == 'repay').sum()
        row['num_liquidations'] = (group['action'] == 'liquidationcall').sum()

        # Handle missing or invalid USD values safely
        usd_values = group['price_usd'].fillna(0) * group['amount'].fillna(0)
        row['total_usd'] = usd_values.sum()
        row['avg_usd'] = usd_values.mean()
        row['usd_stddev'] = usd_values.std()  # ✅ NEW

        # Repay to borrow ratio
        row['repay_borrow_ratio'] = (
            row['num_repays'] / row['num_borrows']
            if row['num_borrows'] > 0 else 0.0
        )

        # Activity duration
        timestamps = group['timestamp'].values
        if len(timestamps) > 1:
            duration = (timestamps[-1] - timestamps[0]) / np.timedelta64(1, 'D')
        else:
            duration = 0.0
        row['activity_duration_days'] = duration
        row['tx_frequency'] = row['tx_count'] / duration if duration > 0 else row['tx_count']  # ✅ NEW

        # ✅ avg_tx_usd (independent check)
        row['avg_tx_usd'] = row['total_usd'] / row['tx_count'] if row['tx_count'] > 0 else 0.0

        features.append(row)

    return pd.DataFrame(features)
