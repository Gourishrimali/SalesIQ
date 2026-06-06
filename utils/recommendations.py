def generate_inventory_recommendation(predicted_total, current_stock_capacity):
    if predicted_total > current_stock_capacity * 1.2:
        return "High demand expected. Strong restock recommended."
    if predicted_total > current_stock_capacity:
        return "Restock recommended. Forecasted demand is higher than current stock capacity."
    if predicted_total > current_stock_capacity * 0.8:
        return "Stock is mostly sufficient, but monitor demand closely."
    return "Stock level is sufficient based on the forecast."