import google.generativeai as genai
import streamlit as st

def generate_business_summary(metrics_dict, anomalies_df=None):
    """
    Takes structural analytics inputs and feeds them to a cloud instance
    of Google Gemini via its official API for production deployment.
    """
    # 1. Retrieve the API key securely from Streamlit's secrets environment
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
    except Exception:
        return """
### ❌ Configuration Error
Streamlit Secrets matrix missing `GEMINI_API_KEY`. 
Please configure your management console secrets to unlock cloud AI processing.
        """

    # 2. Structural prompt generation (Keeps your exact original metrics formatting)
    prompt = f"""
    Analyze the following corporate sales metrics and generate a concise executive summary report.
    Use clear headers, bullet points, and highlight areas of strength or concern.

    Core Metrics:
    - Total Sales Revenue: {metrics_dict.get('total_sales', 'N/A')}
    - Total Orders Processed: {metrics_dict.get('total_orders', 'N/A')}
    - Net Accrued Profit: {metrics_dict.get('total_profit', 'N/A')}
    - Operational Profit Margin: {metrics_dict.get('profit_margin', 'N/A')}
    - Average Order Value (AOV): {metrics_dict.get('aov', 'N/A')}
    - Leading Product Category: {metrics_dict.get('best_category', 'N/A')}
    - Strongest Operational Region: {metrics_dict.get('best_region', 'N/A')}
    - MVP Top Performing Product SKU: {metrics_dict.get('best_product', 'N/A')}
    """

    if anomalies_df is not None and not anomalies_df.empty:
        try:
            anomalies_list = anomalies_df.tail(3).to_dict(orient="records")
            prompt += f"\nCritical Outliers & Anomalies Flagged in Data Matrix:\n{anomalies_list}\n"
        except Exception:
            pass
    
    prompt += "\nProvide a 3-paragraph executive analysis summarizing overall performance, category insights, and a concrete strategic recommendation."

    # 3. Requesting generation from the active Cloud API model
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Cloud AI Generation Error: {str(e)}"