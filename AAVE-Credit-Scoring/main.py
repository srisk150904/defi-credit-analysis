import pandas as pd
import json

def load_json_to_df(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    return df

def extract_features(df):
    df['wallet'] = df['userWallet']
    df['action'] = df['action']
    features = df.groupby('wallet')['action'].value_counts().unstack(fill_value=0)
    return features

def score_wallets(features_df):
    features_df['score'] = (
        features_df.get('repay', 0) * 3 -
        features_df.get('borrow', 0) * 2 +
        features_df.get('deposit', 0)
    )
    features_df['score'] = features_df['score'].clip(lower=0)
    features_df['score_scaled'] = 1000 * (features_df['score'] / features_df['score'].max())
    return features_df[['score_scaled']].rename(columns={'score_scaled': 'credit_score'})

def main():
    df = load_json_to_df("user_transactions.json")
    features = extract_features(df)
    scored = score_wallets(features)
    scored.to_csv("wallet_scores.csv")
    print("Credit scores saved to wallet_scores.csv")

if __name__ == "__main__":
    main()
