import pandas as pd

def generate_labels(df: pd.DataFrame, forward_days: int = 20, bull_thresh: float = 0.03, bear_thresh: float = -0.03) -> pd.DataFrame:
    df = df.copy()
    
    df['forward_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1

    def assign_label(ret):
        if pd.isna(ret):
            return None
        elif ret > bull_thresh:
            return 'Bullish'
        elif ret < bear_thresh:
            return 'Bearish'
        else:
            return 'Sideways'

    df['label'] = df['forward_return'].apply(assign_label)
    
    df.dropna(subset=['label'], inplace=True)
    
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw_data.csv", index_col=0, parse_dates=True)
    labeled_df = generate_labels(df)
    print(labeled_df['label'].value_counts())
    labeled_df.to_csv("data/labeled_data.csv")
    print("Saved labeled data to data/labeled_data.csv")