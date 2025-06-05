import pandas as pd
import numpy as np

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8')

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df['Holiday'] = df['Holiday'].map({'No Holiday': 0, 'Holiday': 1})

    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    df['Seasons'] = pd.Categorical(df['Seasons'], categories=season_order, ordered=True)
    df['Seasons'] = df['Seasons'].cat.codes

    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    drop_cols = ['Date', 'Functioning Day']

    df.drop(columns=drop_cols, inplace=True)

    df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") for col in df.columns]
    return df