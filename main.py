# main.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ✅ Page Config
st.set_page_config(page_title="MarketMind AI", layout="wide")

# ✅ Custom CSS for responsive mobile nav
st.markdown("""
    <style>
    @media (max-width: 768px) {
        .nav-radio label {
            display: block !important;
            margin: 10px 0 !important;
        }
    }
    .nav-radio label {
        display: inline-block;
        margin-right: 20px;
        font-weight: bold;
        color: #00c4b4;
        font-size: 16px;
    }
    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
nav = st.radio("Navigation", ["📊 Dashboard", "👨‍💻 About Developer", "📘 About Project"], horizontal=True, key="nav-radio")

if nav == "👨‍💻 About Developer":
    st.title("👨‍💻 About the Developer")
    st.markdown("""
    ### Hey there! I'm Ansh Mishra ✨

    I'm a 17-year-old aspiring AI/ML engineer and the creative mind behind **MarketMind AI** — my first full-fledged AI-powered marketing analytics dashboard.

    - 🎓 B.Tech (AI & ML) Student at IILM University, Greater Noida
    - 💡 Passionate about AI, Automation, and UI/UX
    - ⚽ Fan of Manchester United | 🚗 Car & Bike Enthusiast | 🦇 Batman Forever
    - 🌐 [GitHub](https://github.com/mishraansh07) | [LinkedIn](https://www.linkedin.com/in/anshmishra007/) | [Instagram](https://instagram.com/mishraansh._)

    _Thank you for checking out my project! I'm open to collaborations, feedback, and creative ideas._
    """)

elif nav == "📘 About Project":
    st.title("📘 About MarketMind AI")
    st.markdown("""
    **MarketMind AI** is a data-driven marketing analytics dashboard that uses AI to:

    - Analyze campaign performance
    - Calculate important KPIs (CTR, CPC, CPA, ROI)
    - Predict future conversions using Machine Learning
    - Give actionable AI-powered suggestions

    ### 🛠️ Built With:
    - `pandas`: For data manipulation
    - `matplotlib`: For creating visualizations
    - `sklearn`: For model training using `RandomForestRegressor`
    - `streamlit`: For the full web app interface

    ### 🚀 How to Use:
    1. Prepare your marketing data in a CSV with these columns:
        - `Campaign`, `Impressions`, `Clicks`, `Spend (₹)`, `Conversions`
    2. Upload the CSV in the dashboard
    3. This tool will:
        - Validate and preview your data
        - Calculate performance KPIs
        - Show improvement suggestions
        - Train a model and predict conversions
        - Allow you to download predictions

    _Perfect for marketing analysts, startups, and digital marketers looking for quick, intelligent insights._
    """)

else:
    st.markdown("""
    <h2 style='text-align: center; color: #00c4b4;'>📊 Welcome to MarketMind AI Dashboard</h2>
    """, unsafe_allow_html=True)

    #loging strating Point

    required_columns = ["Campaign", "Impressions", "Clicks", "Spend (₹)", "Conversions"]
    File_uploader = st.file_uploader("📁 Upload Your Marketing CSV File", type=["csv"])

    if File_uploader is None:
        st.markdown("### 📄 Sample CSV Format")
        sample_data = pd.DataFrame({
            "Campaign": ["Campaign A", "Campaign B", "Campaign C"],
            "Impressions": [10000, 8000, 15000],
            "Clicks": [250, 160, 300],
            "Spend (₹)": [5000, 4000, 7000],
            "Conversions": [30, 18, 25]
        })
        st.dataframe(sample_data)
        csv = sample_data.to_csv(index=False)
        st.download_button("⬇️ Download Sample CSV", data=csv, file_name='sample_marketing_data.csv', mime='text/csv')

    if File_uploader:
        try:
            df = pd.read_csv(File_uploader)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing column(s): {', '.join(missing_cols)}")
                st.stop()

            if df[required_columns].isnull().any().any():
                st.error("❌ Your file contains missing values. Please clean your data.")
                st.stop()

            st.success("✅ File Uploaded and Validated Successfully!")
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df)

            st.subheader("📊 Summary Statistics")
            st.write(df.describe())

            # KPIs
            df["CTR (%)"] = (df["Clicks"] / df["Impressions"]) * 100
            df["CPC (₹)"] = df["Spend (₹)"] / df["Clicks"]
            df["CPA (₹)"] = df["Spend (₹)"] / df["Conversions"]
            df["ROI (%)"] = ((df["Conversions"] * 500 - df["Spend (₹)"]) / df["Spend (₹)"]) * 100

            st.subheader("📈 Calculated KPIs")
            st.dataframe(df[["Campaign", "CTR (%)", "CPC (₹)", "CPA (₹)", "ROI (%)"]])

            col1, col2 = st.columns(2)
            col1.metric("Average CTR (%)", f"{df['CTR (%)'].mean():.2f}")
            col2.metric("Average ROI (%)", f"{df['ROI (%)'].mean():.2f}")

            selected_campaign = st.selectbox(
                "🔍 Select a campaign to view insights:",
                options=["All Campaigns"] + df["Campaign"].unique().tolist()
            )
            filtered_df = df if selected_campaign == "All Campaigns" else df[df["Campaign"] == selected_campaign]

            if selected_campaign != "All Campaigns":
                st.subheader("🧠 AI-Powered Insights & Suggestions")
                shown_campaigns = set()
                for _, row in filtered_df.iterrows():
                    campaign_name = row["Campaign"]
                    if campaign_name in shown_campaigns:
                        continue
                    shown_campaigns.add(campaign_name)

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
                        <div style="background-color: #111; color: white; padding: 15px; border-radius: 10px;
                        margin-bottom: 10px; border-left: 5px solid #00c4b4;">
                        <strong>{campaign_name}:</strong><br>{suggestion}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Train model & predict
            features = ["Impressions", "Clicks", "Spend (₹)", "CTR (%)", "CPC (₹)", "CPA (₹)"]
            target = "Conversions"

            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            df["Predicted Conversions"] = model.predict(X)
            st.subheader("🤖 Predicted Conversions")
            st.dataframe(df[["Campaign", "Conversions", "Predicted Conversions"]])

            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=df.to_csv(index=False),
                file_name='predictions.csv',
                mime='text/csv'
            )

            # Bar chart
            st.subheader("📊 Conversions per Campaign")
            campaign_group = df.groupby("Campaign")["Conversions"].sum().sort_values()
            fig, ax = plt.subplots(figsize=(10, 4))
            campaign_group.plot(kind="barh", ax=ax, color="#00c4b4")
            ax.set_xlabel("Total Conversions")
            ax.set_title("Campaign Performance")
            st.pyplot(fig)

            # Explore individual campaign
            st.subheader("🔍 Explore Individual Campaign")
            individual_campaign = st.selectbox("Select a Campaign", df["Campaign"].unique())
            st.write(df[df["Campaign"] == individual_campaign])

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
    else:
        st.info("Please upload a CSV file to continue.")
