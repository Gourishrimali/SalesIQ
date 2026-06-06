from google import genai


def create_sales_context(df):
    total_sales = df["Sales"].sum()
    total_orders = len(df)
    average_sales = df["Sales"].mean()

    context = []
    context.append(f"Total orders: {total_orders}")
    context.append(f"Total sales revenue: ${total_sales:,.2f}")
    context.append(f"Average sales per order: ${average_sales:,.2f}")

    if "Profit" in df.columns:
        total_profit = df["Profit"].sum()
        profit_margin = total_profit / total_sales if total_sales > 0 else 0
        context.append(f"Total profit: ${total_profit:,.2f}")
        context.append(f"Profit margin: {profit_margin * 100:.2f}%")

    if "Category" in df.columns:
        category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
        context.append(f"Best category: {category_sales.index[0]}")
        context.append(f"Best category sales: ${category_sales.iloc[0]:,.2f}")

    if "Region" in df.columns:
        region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
        context.append(f"Best region: {region_sales.index[0]}")
        context.append(f"Best region sales: ${region_sales.iloc[0]:,.2f}")

    if "Product Name" in df.columns:
        product_sales = df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False)
        context.append(f"Best product: {product_sales.index[0]}")
        context.append(f"Best product sales: ${product_sales.iloc[0]:,.2f}")

    return "\n".join(context)


def generate_business_summary(df, api_key=None):
    context = create_sales_context(df)

    if not api_key:
        return (
            "Gemini API key not found.\n\n"
            "Please add GEMINI_API_KEY in .streamlit/secrets.toml.\n\n"
            f"Sales Context:\n{context}"
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a professional business analyst.

Analyze the following sales data summary and write a clear business report.

Sales data summary:
{context}

Start the report with this exact line:
"Generated using Google Gemini AI for SalesIQ."

Write the report with:
1. Overall sales performance
2. Profitability analysis
3. Best-performing category, region, and product
4. Risks or concerns
5. Recommended business actions

Keep the report practical, professional, and easy to understand.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as error:
        return (
            "Error while generating Gemini AI report.\n\n"
            f"Error details: {error}\n\n"
            f"Sales Context:\n{context}"
        )