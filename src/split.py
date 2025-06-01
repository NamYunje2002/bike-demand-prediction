import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.2, random_state=42):
    train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train_val, test_size=val_size, random_state=random_state)

    return train, val, test