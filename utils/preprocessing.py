import pandas as pd

def clean_sales_data(df):
    df = df.copy()

    df.columns = df.columns.str.strip()

    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    numeric_columns = ["Sales", "Profit", "Quantity"]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Order Date" in df.columns and "Sales" in df.columns:
        df = df.dropna(subset=["Order Date", "Sales"])

    return df
    