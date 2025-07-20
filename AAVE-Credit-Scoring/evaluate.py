import pandas as pd
import matplotlib.pyplot as plt

def plot_score_distribution(scored_df: pd.DataFrame, save_path=None):
    """
    Plots histogram of credit scores using matplotlib only.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scored_df['credit_score'], bins=20, color='skyblue', edgecolor='black')
    plt.title("Credit Score Distribution")
    plt.xlabel("Credit Score")
    plt.ylabel("Number of Wallets")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_bucket_summary(scored_df: pd.DataFrame):
    """
    Prints how many wallets fall into each credit score bucket (0–1000 in steps of 100).
    """
    scored_df['bucket'] = pd.cut(scored_df['credit_score'], bins=[i for i in range(0, 1100, 100)])
    print("✅ Credit Score Buckets:")
    print(scored_df['bucket'].value_counts().sort_index())

def show_wallet_examples(scored_df: pd.DataFrame, full_features_df: pd.DataFrame, top_n=3):
    """
    Displays wallets with highest, median, and lowest scores along with their key behaviors.
    """
    merged = scored_df.merge(full_features_df, on='wallet_address', how='left')

    sorted_df = merged.sort_values('credit_score')

    print("\n🔴 Lowest Scoring Wallets")
    print(sorted_df[['wallet_address', 'credit_score', 'num_deposits',
                     'repay_borrow_ratio', 'total_usd', 'activity_duration_days']].head(top_n))

    print("\n🟡 Median Scoring Wallets")
    median_score = sorted_df['credit_score'].median()
    median_df = sorted_df.iloc[(sorted_df['credit_score'] - median_score).abs().argsort()[:top_n]]
    print(median_df[['wallet_address', 'credit_score', 'num_deposits',
                     'repay_borrow_ratio', 'total_usd', 'activity_duration_days']])

    print("\n🟢 Highest Scoring Wallets")
    print(sorted_df[['wallet_address', 'credit_score', 'num_deposits',
                     'repay_borrow_ratio', 'total_usd', 'activity_duration_days']].tail(top_n))
