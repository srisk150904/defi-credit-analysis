# src/parse_data.py

import json
import pandas as pd

def load_json(file_path: str) -> list:
    """
    Loads a JSON file containing transaction data.
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_to_dataframe(data: list) -> pd.DataFrame:
    """
    Converts raw AAVE JSON records to a structured pandas DataFrame.
    Normalizes nested fields like 'actionData'.
    """
    df = pd.json_normalize(data)

    # Rename and convert timestamp
    df = df.rename(columns={"userWallet": "wallet_address"})
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Flatten actionData fields (optional cleanup)
    if 'actionData.amount' in df.columns:
        df['amount'] = pd.to_numeric(df['actionData.amount'], errors='coerce')
    if 'actionData.assetSymbol' in df.columns:
        df['asset'] = df['actionData.assetSymbol']
    if 'actionData.assetPriceUSD' in df.columns:
        df['price_usd'] = pd.to_numeric(df['actionData.assetPriceUSD'], errors='coerce')

    return df[['wallet_address', 'action', 'timestamp', 'amount', 'asset', 'price_usd']].copy()
