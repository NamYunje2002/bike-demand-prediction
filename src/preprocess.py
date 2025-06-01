import pandas as pd

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8')

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    df = df[df['Functioning Day'] == 'Yes']

    df['Holiday'] = df['Holiday'].map({'No Holiday': 0, 'Holiday': 1})

    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    df['Seasons'] = pd.Categorical(df['Seasons'], categories=season_order, ordered=True)
    df['Seasons'] = df['Seasons'].cat.codes

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday

    drop_cols = ['Date', 'Functioning Day', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)']

    df.drop(columns=drop_cols, inplace=True)

    df.columns = [col.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") for col in df.columns]

    return df