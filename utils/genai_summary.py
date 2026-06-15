import json
import requests

def generate_business_summary(metrics_dict, anomalies_df=None):
    """
    Takes structural analytics inputs and feeds them to a local instance
    of Ollama via a streaming HTTP request to bypass deep-thinking timeouts.
    """
    
    # 1. Structural prompt generation
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

    # 2. Direct HTTP call to Ollama endpoint
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "qwen3:4b",
        "messages": [
            {
                "role": "system", 
                "content": "You are an advanced, data-driven executive business intelligence analyst. Provide your response directly without thinking tags."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "options": {
            "temperature": 0.1
        },
        "stream": True # Streaming keeps the network connection alive and kicking!
    }

    try:
        # Activating a streaming response loop 
        response = requests.post(url, json=payload, stream=True, timeout=120.0)
        
        if response.status_code == 200:
            full_response_text = ""
            # Read incoming data line by line as Ollama generates it
            for line in response.iter_lines():
                if line:
                    decoded_line = json.loads(line.decode('utf-8'))
                    # Append text chunk
                    if 'message' in decoded_line and 'content' in decoded_line['message']:
                        full_response_text += decoded_line['message']['content']
            
            if full_response_text.strip():
                return full_response_text
            else:
                raise Exception("Ollama connected successfully but returned an empty text payload.")
        else:
            raise Exception(f"Ollama server responded with error code {response.status_code}: {response.text}")

    except Exception as e:
        error_msg = str(e)
        return f"""
### 📋 Local Dashboard Metric Report (Fallback)

* **Revenue Metrics**: Total operations pulled a volume of {metrics_dict.get('total_sales', 'N/A')} across {metrics_dict.get('total_orders', 'N/A')} orders.
* **Profitability**: Net yield values at {metrics_dict.get('total_profit', 'N/A')} reflecting an active margin of {metrics_dict.get('profit_margin', 'N/A')}.
* **Top Performers**: Growth was heavily anchored by the {metrics_dict.get('best_category', 'N/A')} category inside the {metrics_dict.get('best_region', 'N/A')} region.

---
⚠️ **Ollama Connection Debug Info:**
`{error_msg}`

*(Please verify Ollama is fully responsive in your system tray or terminal.)*
"""