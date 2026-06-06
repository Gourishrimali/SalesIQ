import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import clean_sales_data
from utils.forecasting import forecast_sales, compare_forecast_models
from utils.anomaly import detect_anomalies
from utils.insights import generate_insights
from utils.genai_summary import generate_business_summary
from utils.recommendations import generate_inventory_recommendation
from utils.pdf_report import create_pdf_report

st.set_page_config(
    page_title="SalesIQ",
    page_icon="bar_chart",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #eef6ff 0%, #f8fafc 35%, #ffffff 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #0b2545;
        font-weight: 800;
    }

    h2, h3 {
        color: #12355b;
    }

    .app-subtitle {
        color: #3b5875;
        font-size: 16px;
        margin-bottom: 20px;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
        border: 1px solid #c7e8ef;
        padding: 18px;
        border-radius: 8px;
        box-shadow: 0 4px 14px rgba(11, 37, 69, 0.08);
    }

    [data-testid="stMetricLabel"] {
        font-size: 15px;
        color: #31516f;
    }

    [data-testid="stMetricValue"] {
        font-size: 25px;
        color: #0b2545;
        font-weight: 800;
    }

    .welcome-card {
        background: linear-gradient(135deg, #ffffff 0%, #eef6ff 100%);
        border: 1px solid #c7dff5;
        border-radius: 8px;
        padding: 28px;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 6px 18px rgba(11, 37, 69, 0.08);
    }

    .welcome-card h3 {
        margin-top: 0;
        color: #0b2545;
    }

    .welcome-card p {
        color: #3b5875;
        font-size: 16px;
        line-height: 1.6;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b2545;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff;
    }

    div[data-baseweb="tab-list"] {
        gap: 8px;
    }

    button[data-baseweb="tab"] {
        background-color: #e6f4f1;
        border-radius: 8px;
        padding: 8px 14px;
        color: #0b2545;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #14b8a6;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("SalesIQ: AI-Powered Sales Analytics And Forecasting Platform")

st.markdown(
    """
    <div class="app-subtitle">
    Upload sales data, explore business performance, forecast future sales,
    detect anomalies, and generate AI-powered business reports.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("### Upload Sales Dataset")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a sales dataset containing fields like Order Date, Sales, Profit, Category, Region, and Product Name."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")
    df = clean_sales_data(df)

    required_columns = ["Order Date", "Sales"]

    missing_required_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_required_columns:
        st.error("This dataset must contain Order Date and Sales columns.")
        st.stop()

    filtered_df = df.copy()

    st.sidebar.title("SalesIQ Controls")
    st.sidebar.caption("Use filters to refine the dashboard analysis.")

    if "Order Date" in df.columns:
        min_date = df["Order Date"].min().date()
        max_date = df["Order Date"].max().date()

        selected_date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(selected_date_range) == 2:
            start_date, end_date = selected_date_range
            filtered_df = filtered_df[
                (filtered_df["Order Date"].dt.date >= start_date) &
                (filtered_df["Order Date"].dt.date <= end_date)
            ]

    if "Region" in df.columns:
        selected_regions = st.sidebar.multiselect(
            "Region",
            options=df["Region"].dropna().unique(),
            default=df["Region"].dropna().unique()
        )
        filtered_df = filtered_df[filtered_df["Region"].isin(selected_regions)]

    if "Category" in df.columns:
        selected_categories = st.sidebar.multiselect(
            "Category",
            options=df["Category"].dropna().unique(),
            default=df["Category"].dropna().unique()
        )
        filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Overview",
        "Analytics",
        "Forecasting",
        "Anomalies",
        "Insights",
        "AI Report",
        "About Project"
    ])

    with tab1:
        st.subheader("Business Overview")

        total_sales = filtered_df["Sales"].sum()
        total_orders = len(filtered_df)
        average_order_value = filtered_df["Sales"].mean()

        if "Profit" in filtered_df.columns:
            total_profit = filtered_df["Profit"].sum()
            profit_margin = total_profit / total_sales if total_sales > 0 else 0
        else:
            total_profit = 0
            profit_margin = 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Sales", f"${total_sales:,.2f}")
        col2.metric("Total Profit", f"${total_profit:,.2f}")
        col3.metric("Total Orders", f"{total_orders:,}")
        col4.metric("Avg Order Value", f"${average_order_value:,.2f}")

        st.subheader("Executive Summary")

        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

        if "Category" in filtered_df.columns:
            best_category = filtered_df.groupby("Category")["Sales"].sum().idxmax()
        else:
            best_category = "N/A"

        if "Region" in filtered_df.columns:
            best_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
        else:
            best_region = "N/A"

        if "Product Name" in filtered_df.columns:
            best_product = filtered_df.groupby("Product Name")["Sales"].sum().idxmax()
        else:
            best_product = "N/A"

        summary_col1.metric("Best Category", best_category)
        summary_col2.metric("Best Region", best_region)
        summary_col3.metric("Profit Margin", f"{profit_margin * 100:.2f}%")
        summary_col4.metric(
            "Top Product",
            best_product[:22] + "..." if len(best_product) > 22 else best_product
        )

        st.subheader("Dataset Preview")
        st.dataframe(filtered_df.head(), use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Filtered Data",
            data=csv_data,
            file_name="salesiq_filtered_data.csv",
            mime="text/csv"
        )

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.subheader("Dataset Information")
            st.write("Rows:", filtered_df.shape[0])
            st.write("Columns:", filtered_df.shape[1])

            if "Order Date" in filtered_df.columns:
                st.write("Start Date:", filtered_df["Order Date"].min())
                st.write("End Date:", filtered_df["Order Date"].max())

        with info_col2:
            st.subheader("Data Quality")
            missing_values = filtered_df.isnull().sum()
            missing_values = missing_values[missing_values > 0]

            if len(missing_values) > 0:
                st.write(missing_values)
            else:
                st.success("No missing values found in the filtered data.")

    with tab2:
        st.subheader("Sales Analytics")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            monthly_sales = (
                filtered_df.set_index("Order Date")
                .resample("ME")["Sales"]
                .sum()
                .reset_index()
            )

            fig = px.line(
                monthly_sales,
                x="Order Date",
                y="Sales",
                title="Monthly Sales Trend",
                markers=True,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            if "Category" in filtered_df.columns:
                category_sales = (
                    filtered_df.groupby("Category")["Sales"]
                    .sum()
                    .reset_index()
                )

                fig = px.bar(
                    category_sales,
                    x="Category",
                    y="Sales",
                    title="Category-wise Sales",
                    color="Category",
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if "Region" in filtered_df.columns:
                region_sales = (
                    filtered_df.groupby("Region")["Sales"]
                    .sum()
                    .reset_index()
                )

                fig = px.pie(
                    region_sales,
                    names="Region",
                    values="Sales",
                    title="Region-wise Sales",
                    hole=0.4
                )

                st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            if "Product Name" in filtered_df.columns:
                top_products = (
                    filtered_df.groupby("Product Name")["Sales"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )

                fig = px.bar(
                    top_products,
                    x="Sales",
                    y="Product Name",
                    orientation="h",
                    title="Top 10 Products by Sales",
                    template="plotly_white"
                )

                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Sales Forecasting")

        model_results, evaluation_df = compare_forecast_models(filtered_df)

        if model_results is not None:
            st.subheader("Model Comparison")
            st.dataframe(model_results, use_container_width=True)

            fig = px.line(
                evaluation_df,
                x="Date",
                y=["Actual Sales", "Linear Regression", "Random Forest"],
                title="Actual Sales vs Model Predictions",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data available for model comparison.")

        forecast_days = st.slider("Select forecast period in days", 7, 90, 30)

        selected_model = st.selectbox(
            "Select forecasting model",
            ["Random Forest", "Linear Regression"]
        )

        current_stock_capacity = st.number_input(
            "Enter current stock capacity / demand capacity",
            min_value=0,
            value=100000
        )

        if st.button("Generate Future Forecast"):
            forecast_df = forecast_sales(filtered_df, forecast_days, selected_model)

            fig = px.line(
                forecast_df,
                x="Date",
                y="Predicted Sales",
                title=f"Future Sales Forecast using {selected_model}",
                markers=True,
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

            predicted_total = forecast_df["Predicted Sales"].sum()
            st.success(
                f"Predicted sales for next {forecast_days} days: ${predicted_total:,.2f}"
            )

            recommendation = generate_inventory_recommendation(
                predicted_total,
                current_stock_capacity
            )

            st.info(recommendation)

    with tab4:
        st.subheader("Sales Anomaly Detection")

        anomaly_df = detect_anomalies(filtered_df)

        fig = px.scatter(
            anomaly_df,
            x="Order Date",
            y="Sales",
            color="Anomaly",
            title="Sales Anomaly Detection",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        anomalies = anomaly_df[anomaly_df["Anomaly"] == True]

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies, use_container_width=True)

    with tab5:
        st.subheader("Business Insights")

        insights = generate_insights(filtered_df)

        for insight in insights:
            st.info(insight)

    with tab6:
        st.subheader("Gemini AI Business Analyst Report")

        st.success("This report is generated using Google Gemini AI.")

        api_key = st.secrets.get("GEMINI_API_KEY", None)
        business_summary = generate_business_summary(filtered_df, api_key)

        st.markdown("### Generated Report")
        st.markdown(business_summary)

        pdf_report = create_pdf_report(business_summary)

        st.download_button(
            label="Download Gemini AI Report as PDF",
            data=pdf_report,
            file_name="salesiq_gemini_ai_report.pdf",
            mime="application/pdf"
        )

        st.download_button(
            label="Download Gemini AI Report as Text",
            data=business_summary,
            file_name="salesiq_gemini_ai_report.txt",
            mime="text/plain"
        )

    with tab7:
        st.subheader("About SalesIQ")

        st.write(
            "SalesIQ is an AI-powered sales analytics and forecasting platform "
            "designed to convert raw sales data into business decisions."
        )

        st.markdown(
            """
            **Problem Statement:**  
            Businesses often have sales data but lack tools to forecast demand,
            detect unusual patterns, and generate decision-ready reports.

            **Tech Stack:**  
            Python, Streamlit, Pandas, Plotly, Scikit-learn, Google Gemini API.

            **Models Used:**  
            Linear Regression and Random Forest Regressor for sales forecasting.

            **GenAI Integration:**  
            Google Gemini AI generates a professional business analyst report from sales metrics.

            **Main Modules:**  
            Dashboard, analytics, forecasting, anomaly detection, business insights,
            Gemini AI report, and inventory recommendation.
            """
        )

else:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>Get Started With SalesIQ</h3>
            <p>
                Upload a sales CSV file to generate an interactive dashboard,
                sales forecast, anomaly detection results, business insights,
                and an AI-powered analyst report.
            </p>
            <p>
                Recommended columns:
                <b>Order Date</b>, <b>Sales</b>, <b>Profit</b>, <b>Quantity</b>,
                <b>Category</b>, <b>Region</b>, and <b>Product Name</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dashboard", "KPIs + Charts")

    with col2:
        st.metric("ML Forecast", "2 Models")

    with col3:
        st.metric("AI Report", "Gemini API")

    st.markdown("### What SalesIQ Will Generate")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.write("- Business performance overview")
        st.write("- Monthly sales trend")
        st.write("- Category and region analysis")
        st.write("- Top product ranking")
        st.write("- Model comparison")

    with feature_col2:
        st.write("- Future sales forecasting")
        st.write("- Sales anomaly detection")
        st.write("- Inventory recommendation")
        st.write("- Gemini-powered AI report")
        st.write("- PDF and text report downloads")