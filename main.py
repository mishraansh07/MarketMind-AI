# main.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="MarketMind AI", layout="wide")

# Custom CSS for responsive mobile nav
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
nav = st.radio("Navigation", ["📊 Dashboard", "📈 Visual Insights", "✨ What's New", "👨‍💻 About Developer", "📘 About Project", "🛡️ License & Disclaimer"], horizontal=True, key="nav-radio")

if nav == "👨‍💻 About Developer":
    st.title("👨‍💻 About the Developer")
    st.markdown("""
    ### Hey there! I'm Ansh Mishra ✨

    I'm a 17-year-old aspiring AI/ML engineer and the creative mind behind **MarketMind AI** — my first full-fledged AI-powered marketing analytics dashboard.

    - 🎓 B.Tech (AI & ML) Student at IILM University, Greater Noida
    - 💡 Passionate about AI, Automation, and UI/UX
    - ⚽ Fan of Manchester United | 🚗 Car & Bike Enthusiast | 🦇 Batman Forever
    - 📧 Contact: ansh.mishra22@outlook.com
    - 🌐 [GitHub](https://github.com/mishraansh07) | [LinkedIn](https://www.linkedin.com/in/anshmishra007/) | [Instagram](https://instagram.com/mishraansh._)

    _Thank you for checking out my project! I'm open to collaborations, feedback, and creative ideas._
    """)

elif nav == "📘 About Project":
    st.title("📘 About MarketMind AI")
    st.markdown("""
    **MarketMind AI** is a cutting-edge analytics dashboard designed to turn raw marketing data into meaningful, predictive insights — powered by AI.

    ### 🚀 Features:
    - Upload one or more CSVs for instant marketing analysis
    - Automatically calculates KPIs: CTR, CPC, CPA, ROI
    - Predict future conversions using ML
    - View top/bottom performing campaigns
    - Interactive visual analytics and downloadable reports

    ### ⚙️ Tech Stack:
    - Python, Pandas, Streamlit
    - Machine Learning (Scikit-Learn)
    - Visualization (Matplotlib & Seaborn)

    _Built with precision for marketers, analysts, and data-driven decision makers._
    """)

elif nav == "🛡️ License & Disclaimer":
    st.title("🛡️ License & Disclaimer")
    st.markdown("""
    ### License
    This project is licensed under the MIT License. You are free to use, modify, and distribute it as you wish, but attribution is appreciated.

    ### Disclaimer ⚠️
    > MarketMind AI is an educational and experimental tool. It is not 100% accurate and should **not** be relied upon for critical business decisions. The insights and predictions are for **personal insight and reference only**.
    """)

elif nav == "✨ What's New":
    st.title("✨ What's New in MarketMind AI")
    st.markdown("""
    - ✅ Multi-file CSV upload and automatic merging
    - 📊 Choose from multiple graph types in 'Visual Insights'
    - 🏆 Highlights best and worst campaign performance
    - 🧠 AI-based improvement suggestions
    - 🎯 Refreshed UI with sleek modern theme and minimal charts

    ---
    ### 🔧 Planned Updates
    - ⏳ Export graphs as PNG or SVG
    - 🗃️ Integrated database support for storing campaigns
    - 📅 Time-series trend tracking by campaign
    - 🤖 Enhanced AI suggestions using GPT-powered insight engine
    - 🌓 Dark/Light Mode Toggle
    - 🧮 Custom KPI Weightage Adjuster
    """)

elif nav == "📈 Visual Insights":
    st.title("📈 Visual Insights Dashboard")
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload and analyze data in the Dashboard tab first.")
    else:
        df = st.session_state.df
        chart_type = st.selectbox("Choose a chart to view:", [
            "Conversions per Campaign",
            "Spend vs Conversions",
            "ROI Distribution",
            "CTR by Campaign",
            "CPC vs CPA",
            "Impressions vs Clicks"
        ])

        fig, ax = plt.subplots(figsize=(8, 4))
        if chart_type == "Conversions per Campaign":
            df.groupby("Campaign")["Conversions"].sum().plot(kind="bar", ax=ax, color="#00c4b4")
            ax.set_title("Conversions per Campaign")

        elif chart_type == "Spend vs Conversions":
            sns.scatterplot(data=df, x="Spend (₹)", y="Conversions", hue="Campaign", ax=ax, palette="viridis")
            ax.set_title("Spend vs Conversions")

        elif chart_type == "ROI Distribution":
            sns.histplot(df["ROI (%)"], kde=True, ax=ax, color="#00c4b4")
            ax.set_title("Distribution of ROI")

        elif chart_type == "CTR by Campaign":
            sns.barplot(x="CTR (%)", y="Campaign", data=df, ax=ax, palette="crest")
            ax.set_title("CTR by Campaign")

        elif chart_type == "CPC vs CPA":
            sns.scatterplot(data=df, x="CPC (₹)", y="CPA (₹)", hue="Campaign", ax=ax, palette="rocket")
            ax.set_title("CPC vs CPA")

        elif chart_type == "Impressions vs Clicks":
            sns.lineplot(data=df, x="Impressions", y="Clicks", hue="Campaign", ax=ax, palette="mako")
            ax.set_title("Impressions vs Clicks")

        st.pyplot(fig)

else:
    st.markdown("""
    <h2 style='text-align: center; color: #00c4b4;'>📊 Welcome to MarketMind AI Dashboard</h2>
    """, unsafe_allow_html=True)

    required_columns = ["Campaign", "Impressions", "Clicks", "Spend (₹)", "Conversions"]
    uploaded_files = st.file_uploader("📁 Upload One or More CSV Files", type=["csv"], accept_multiple_files=True)

    st.markdown("""
    #### 📌 Note
    ⚠️ *You can now upload multiple CSV files.* However, for optimal performance and better insights:
    - Upload a **reasonable number of campaigns per file**
    - Prefer **10–50 total campaigns** for clean graphs and accurate AI predictions
    - Very large datasets may slow down performance or clutter visualizations
    """)

    if not uploaded_files:
        st.markdown("### 📄 Sample CSV Format")
        sample_data = pd.DataFrame({
            "Campaign": ["Campaign A", "Campaign B", "Campaign C"],
            "Impressions": [10000, 8000, 15000],
            "Clicks": [250, 160, 300],
            "Spend (₹)": [5000, 4000, 7000],
            "Conversions": [30, 18, 25]
        })
        sample_data.index += 1
        st.dataframe(sample_data)
        csv = sample_data.to_csv(index=False)
        st.download_button("⬇️ Download Sample CSV", data=csv, file_name='sample_marketing_data.csv', mime='text/csv')

    all_dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"❌ '{file.name}' Missing column(s): {', '.join(missing_cols)}")
                continue
            if df[required_columns].isnull().any().any():
                st.error(f"❌ '{file.name}' contains missing values. Please clean your data.")
                continue
            all_dfs.append(df)
        except Exception as e:
            st.error(f"❌ Error in '{file.name}': {e}")

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        df.index += 1

        df["CTR (%)"] = (df["Clicks"] / df["Impressions"]) * 100
        df["CPC (₹)"] = df["Spend (₹)"] / df["Clicks"]
        df["CPA (₹)"] = df["Spend (₹)"] / df["Conversions"]
        df["ROI (%)"] = ((df["Conversions"] * 500 - df["Spend (₹)"]) / df["Spend (₹)"]) * 100

        st.session_state.df = df

        st.success("✅ Data Uploaded and Processed Successfully!")
        st.subheader("📋 Uploaded Data Preview")
        st.dataframe(df)

        st.subheader("📊 Summary Statistics")
        st.write(df.describe())

        st.subheader("📈 Calculated KPIs")
        st.dataframe(df[["Campaign", "CTR (%)", "CPC (₹)", "CPA (₹)", "ROI (%)"]])

        col1, col2 = st.columns(2)
        col1.metric("Average CTR (%)", f"{df['CTR (%)'].mean():.2f}")
        col2.metric("Average ROI (%)", f"{df['ROI (%)'].mean():.2f}")

        st.markdown("### 🏆 Campaign Performance Rankings")
        best = df.loc[df["ROI (%)"].idxmax()]['Campaign']
        worst = df.loc[df["ROI (%)"].idxmin()]['Campaign']
        st.success(f"✅ Best Performing Campaign: **{best}**")
        st.error(f"❌ Worst Performing Campaign: **{worst}**")

        selected_campaign = st.selectbox("🔍 Select a campaign to view insights:", options=["All Campaigns"] + df["Campaign"].unique().tolist())
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
                st.markdown(f"""
                    <div style="background-color: #111; color: white; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #00c4b4;">
                    <strong>{campaign_name}:</strong><br>{suggestion}</div>
                """, unsafe_allow_html=True)

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

        st.download_button("⬇️ Download Predictions as CSV", data=df.to_csv(index=False), file_name='predictions.csv', mime='text/csv')

        st.subheader("🔍 Explore Individual Campaign")
        individual_campaign = st.selectbox("Select a Campaign", df["Campaign"].unique())
        st.write(df[df["Campaign"] == individual_campaign])
    else:
        st.info("Please upload one or more CSV files to continue.")

