# src/model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from utils import scale_score_to_range


MODEL_PATH = "model.pkl"

def train_model(feature_df: pd.DataFrame, save_path: str = MODEL_PATH) -> RandomForestRegressor:
    """
    Trains a Random Forest model to predict a pseudo credit score.
    Applies log and clipping to stabilize feature scales.
    """
    df = feature_df.copy()

    # Stabilize exploding features
    df['log_total_usd'] = np.log1p(df['total_usd'])           # log(1 + total_usd)
    df['clipped_ratio'] = df['repay_borrow_ratio'].clip(0, 5) # clip extreme outliers

    # Define features
    X = df[[
        'num_deposits',
        'num_borrows',
        'num_repays',
        'num_liquidations',
        'log_total_usd',
        'clipped_ratio',
        'activity_duration_days'
    ]]

    y = (
        df['num_deposits'] * 1.1
        + df['repay_borrow_ratio'].clip(0, 5) * 5  # reduce outlier impact
        - df['num_liquidations'] * 37
        + df['activity_duration_days'] * 0.2
        + df['total_usd'].apply(np.log1p) / 1000
        + np.log1p(df['avg_tx_usd']) / 100
    )

    y = scale_score_to_range(y, 0, 1000)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    dump(model, save_path)
    return model


def score_wallets(model: RandomForestRegressor, feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()
    df['log_total_usd'] = np.log1p(df['total_usd'])
    df['clipped_ratio'] = df['repay_borrow_ratio'].clip(0, 5)

    X = df[[
        'num_deposits',
        'num_borrows',
        'num_repays',
        'num_liquidations',
        'log_total_usd',
        'clipped_ratio',
        'activity_duration_days'
    ]]

    predictions = model.predict(X)
    predictions = scale_score_to_range(predictions, 0, 1000)

    result_df = feature_df[['wallet_address']].copy()
    result_df['credit_score'] = predictions.astype(int)
    return result_df



# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# from lightgbm import LGBMRegressor
# from sklearn.preprocessing import MinMaxScaler
# from joblib import dump, load
# from utils import scale_score_to_range

# MODEL_PATH = "model.pkl"

# def train_model(feature_df: pd.DataFrame, save_path: str = MODEL_PATH) -> RandomForestRegressor:
#     """
#     Trains a Random Forest model to predict a pseudo credit score.
#     Adds new features: avg_tx_usd, usd_stddev, tx_frequency.
#     """
#     df = feature_df.copy()

#     # Log-transform + clip for stability
#     df['log_total_usd'] = np.log1p(df['total_usd'])
#     df['clipped_ratio'] = df['repay_borrow_ratio'].clip(0, 5)

#     # Define features to use
#     X = df[[
#         'num_deposits',
#         'num_borrows',
#         'num_repays',
#         'num_liquidations',
#         'log_total_usd',
#         'clipped_ratio',
#         'activity_duration_days',
#         'avg_tx_usd',
#         'usd_stddev',
#         'tx_frequency',
#     ]]

#     # Improved pseudo label
#     y = (
#         df['num_deposits'] * 3
#         + df['clipped_ratio'] * 8
#         - df['num_liquidations'] * 15
#         + df['activity_duration_days'] * 0.5
#         + df['log_total_usd'] / 5_000
#         + df['tx_frequency'] * 0.25
#     )

#     y = scale_score_to_range(y, 0, 1000)

#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X, y)
#     dump(model, save_path)
#     return model


# def score_wallets(model: RandomForestRegressor, feature_df: pd.DataFrame) -> pd.DataFrame:
#     df = feature_df.copy()

#     # Same transformation steps as training
#     df['log_total_usd'] = np.log1p(df['total_usd'])
#     df['clipped_ratio'] = df['repay_borrow_ratio'].clip(0, 5)

#     X = df[[
#         'num_deposits',
#         'num_borrows',
#         'num_repays',
#         'num_liquidations',
#         'log_total_usd',
#         'clipped_ratio',
#         'activity_duration_days',
#         'avg_tx_usd',
#         'usd_stddev',
#         'tx_frequency',
#     ]]

#     predictions = model.predict(X)
#     predictions = scale_score_to_range(predictions, 0, 1000)

#     result_df = feature_df[['wallet_address']].copy()
#     result_df['credit_score'] = predictions.astype(int)
#     return result_df




# import pandas as pd
# import numpy as np
# from lightgbm import LGBMRegressor
# from sklearn.preprocessing import MinMaxScaler
# from joblib import dump, load
# from utils import scale_score_to_range

# MODEL_PATH = "model_lgb.pkl"

# def train_model(feature_df: pd.DataFrame, save_path: str = MODEL_PATH) -> LGBMRegressor:
#     """
#     Trains a LightGBM model to predict a pseudo credit score using engineered wallet features.
#     """
#     df = feature_df.copy()
#     df['log_total_usd'] = np.log1p(df['total_usd'])
#     df['clipped_ratio'] = df['repay_borrow_ratio'].clip(0, 5)

#     X = df[[
#         'num_deposits',
#         'num_borrows',
#         'num_repays',
#         'num_liquidations',
#         'log_total_usd',
#         'clipped_ratio',
#         'activity_duration_days',
#         'avg_tx_usd',
#         'usd_stddev',
#         'tx_frequency',
#     ]]

#     y = (
#         df['num_deposits'] * 3
#         + df['clipped_ratio'] * 8
#         - df['num_liquidations'] * 15
#         + df['activity_duration_days'] * 0.5
#         + df['log_total_usd'] / 5000
#         + df['tx_frequency'] * 0.25
#     )
#     y = scale_score_to_range(y, 0, 1000)

#     model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
#     model.fit(X, y)

#     dump(model, save_path)
#     return model


# def score_wallets(model: LGBMRegressor, feature_df: pd.DataFrame) -> pd.DataFrame:
#     df = feature_df.copy()
#     df['log_total_usd'] = np.log1p(df['total_usd'])
#     df['clipped_ratio'] = df['repay_borrow_ratio'].clip(0, 5)

#     X = df[[
#         'num_deposits',
#         'num_borrows',
#         'num_repays',
#         'num_liquidations',
#         'log_total_usd',
#         'clipped_ratio',
#         'activity_duration_days',
#         'avg_tx_usd',
#         'usd_stddev',
#         'tx_frequency',
#     ]]

#     predictions = model.predict(X)
#     predictions = scale_score_to_range(predictions, 0, 1000)

#     result_df = feature_df[['wallet_address']].copy()
#     result_df['credit_score'] = predictions.astype(int)
#     return result_df
