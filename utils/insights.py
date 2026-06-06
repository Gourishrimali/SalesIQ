def generate_insights(df):
    insights = []

    if "Category" in df.columns:
        top_category = df.groupby("Category")["Sales"].sum().idxmax()
        insights.append(f"{top_category} is the highest revenue-generating category.")

    if "Region" in df.columns:
        top_region = df.groupby("Region")["Sales"].sum().idxmax()
        insights.append(f"{top_region} is the best-performing region.")

    if "Profit" in df.columns:
        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()

        if total_sales > 0:
            profit_margin = total_profit / total_sales

            if profit_margin < 0.10:
                insights.append("Profit margin is low. Review discounts, pricing, and operating costs.")
            else:
                insights.append("Profit margin is healthy.")

    if len(insights) == 0:
        insights.append("Not enough data available to generate insights.")

    return insights