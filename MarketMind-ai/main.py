# main.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib as jb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

st.markdown("<h1 style='text-align: center;'>📊 MarketMind AI</h1>",
    unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; font-size:18px;'>Upload your marketing data and get AI-powered insights!</p>",
    unsafe_allow_html=True)



File_uploader = st.file_uploader("Please Uplaod Your File", type=["csv"])

openai = st.secrets["openai"]["api_key"]

#Pandas CSV

if File_uploader is not None:
    df = pd.read_csv(File_uploader)
    st.success("File Uploaded Successfully!")

    #DataFrame
    st.subheader("Uploaded Data Preview: ")
    st.dataframe(df)

    #basic-stats

    st.subheader("Summery Statistics: ")
    st.write(df.describe())

    #Trained-Model

    model = jb.load("model.pkl")

    # Calculate KPIs
    df["CTR (%)"] = (df["Clicks"] / df["Impressions"]) * 100
    df["CPC (₹)"] = df["Spend (₹)"] / df["Clicks"]
    df["CPA (₹)"] = df["Spend (₹)"] / df["Conversions"]
    df["ROI (%)"] = ((df["Conversions"] * 500 - df["Spend (₹)"]) / df["Spend (₹)"]) * 100  # assume 1 conversion = ₹500

# Show updated table
    st.subheader("📈 Calculated KPIs:")
    st.dataframe(df[["Campaign", "CTR (%)", "CPC (₹)", "CPA (₹)", "ROI (%)"]])


    # Step2 
    st.subheader("🧠 AI-Powered Insights & Suggestions")

    for index, row in df.iterrows():
      suggestion = ""

      if row["CTR (%)"] < 2:
        suggestion += "⚠️ Low CTR — try improving your ad creatives. "
      if row["CPA (₹)"] > 300:
        suggestion += "💸 High CPA — optimize your targeting or lower spend. "
      if row["ROI (%)"] < 0:
        suggestion += "📉 Negative ROI — consider stopping or changing the strategy."

      if suggestion == "":
        suggestion = "✅ This campaign is performing well."

    st.markdown(
        f"""
        <div style="
            background-color: #111;
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 5px solid #00c4b4;">
            <strong>{row['Campaign']}:</strong><br>{suggestion}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Step 1: Calculate KPIs ---
    df["CTR (%)"] = (df["Clicks"] / df["Impressions"]) * 100
    df["CPC (₹)"] = df["Spend (₹)"] / df["Clicks"]
    df["CPA (₹)"] = df["Spend (₹)"] / df["Conversions"]
    df["ROI (%)"] = ((df["Conversions"] * 500 - df["Spend (₹)"]) / df["Spend (₹)"]) * 100  # Assuming 1 conversion = ₹500

    st.subheader("📈 Calculated KPIs:")
    st.dataframe(df[["Campaign", "CTR (%)", "CPC (₹)", "CPA (₹)", "ROI (%)"]])

    features = ["Impressions", "Clicks", "Spend (₹)", "CTR (%)", "CPC (₹)", "CPA (₹)"]
    target = "Conversions"

    # Data Prep

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Model training

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    
  
    df["Predicted Conversions"] = model.predict(X)

    st.subheader("🤖 Predicted Conversions (by AI):")
    st.dataframe(df[["Campaign", "Conversions", "Predicted Conversions"]])



    st.download_button(
    label="⬇️ Download Predictions as CSV",
    data=df.to_csv(index=False),
    file_name='predictions.csv',
    mime='text/csv'
)


    st.subheader("🔍 Explore Individual Campaign")
    campaigns = df["Campaign"].unique()
    selected_campaign = st.selectbox("Select a Campaign", campaigns)
    st.write(df[df["Campaign"] == selected_campaign])



    st.subheader("📊 Conversions per Campaign (Bar Chart)")

    campaign_group = df.groupby("Campaign")["Conversions"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    campaign_group.plot(kind="barh", ax=ax, color="#00c4b4")
    ax.set_xlabel("Total Conversions")
    ax.set_title("Campaign Performance")
    st.pyplot(fig)



else:
    st.info("Please upload a CSV file to continue")