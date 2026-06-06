import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_daily_sales(df):
    daily_sales = df.groupby("Order Date")["Sales"].sum().reset_index().sort_values("Order Date")
    daily_sales["day_number"] = np.arange(len(daily_sales))
    daily_sales["day_of_week"] = daily_sales["Order Date"].dt.dayofweek
    daily_sales["month"] = daily_sales["Order Date"].dt.month
    return daily_sales


def get_model(model_name):
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    return LinearRegression()


def forecast_sales(df, days=30, model_name="Random Forest"):
    daily_sales = prepare_daily_sales(df)
    features = ["day_number", "day_of_week", "month"]

    model = get_model(model_name)
    model.fit(daily_sales[features], daily_sales["Sales"])

    future_dates = pd.date_range(
        start=daily_sales["Order Date"].max() + pd.Timedelta(days=1),
        periods=days
    )

    future_df = pd.DataFrame({"Date": future_dates})
    future_df["day_number"] = np.arange(len(daily_sales), len(daily_sales) + days)
    future_df["day_of_week"] = future_df["Date"].dt.dayofweek
    future_df["month"] = future_df["Date"].dt.month

    predictions = model.predict(future_df[features])

    return pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": np.clip(predictions, 0, None)
    })


def evaluate_single_model(train, test, model_name):
    features = ["day_number", "day_of_week", "month"]

    model = get_model(model_name)
    model.fit(train[features], train["Sales"])

    predictions = np.clip(model.predict(test[features]), 0, None)

    mae = mean_absolute_error(test["Sales"], predictions)
    rmse = np.sqrt(mean_squared_error(test["Sales"], predictions))
    mape = np.mean(np.abs((test["Sales"] - predictions) / test["Sales"].replace(0, np.nan))) * 100

    return mae, rmse, mape, predictions


def compare_forecast_models(df):
    daily_sales = prepare_daily_sales(df)

    if len(daily_sales) < 10:
        return None, None

    split_index = int(len(daily_sales) * 0.8)
    train = daily_sales.iloc[:split_index]
    test = daily_sales.iloc[split_index:].copy()

    results = []
    evaluation_df = pd.DataFrame({"Date": test["Order Date"], "Actual Sales": test["Sales"]})

    for model_name in ["Linear Regression", "Random Forest"]:
        mae, rmse, mape, predictions = evaluate_single_model(train, test, model_name)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        })

        evaluation_df[model_name] = predictions

    return pd.DataFrame(results), evaluation_df