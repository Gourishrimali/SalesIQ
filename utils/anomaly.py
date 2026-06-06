def detect_anomalies(df):
    daily_sales = (
        df.groupby("Order Date")["Sales"]
        .sum()
        .reset_index()
        .sort_values("Order Date")
    )

    mean_sales = daily_sales["Sales"].mean()
    std_sales = daily_sales["Sales"].std()

    if std_sales == 0:
        daily_sales["z_score"] = 0
    else:
        daily_sales["z_score"] = (daily_sales["Sales"] - mean_sales) / std_sales

    daily_sales["Anomaly"] = daily_sales["z_score"].abs() > 2

    return daily_sales