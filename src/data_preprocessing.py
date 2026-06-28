import pandas as pd
import numpy as np

def load_data(train_path, store_path):
    df = pd.read_csv(train_path, low_memory=False, parse_dates=["Date"])
    store_df = pd.read_csv(store_path)
    return df, store_df

def clean_data(df, store_df):
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    df = df.merge(store_df, on="Store", how="left")
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)
    return df
