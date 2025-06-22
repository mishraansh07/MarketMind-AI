# 📊 MarketMind AI

MarketMind AI is an intelligent marketing analytics tool that helps businesses understand and optimize their advertising campaigns using AI. Built with Streamlit and machine learning, it lets users upload marketing data in CSV format, calculates key performance indicators (KPIs), predicts conversions, and provides actionable insights.

---

## 🚀 Features

- 📁 Upload your own marketing campaign data as a `.csv` file
- 📈 Auto-calculates important KPIs like:
  - CTR (%)
  - CPC (₹)
  - CPA (₹)
  - ROI (%)
- 🤖 Predicts future conversions using a trained machine learning model
- 💡 Provides simple, readable AI-powered insights and suggestions
- 🖥️ Clean, interactive Streamlit interface

---

## 🧰 Libraries & Technologies Used

| Library         | Purpose                                              |
|----------------|------------------------------------------------------|
| **Streamlit**  | Build the interactive web UI                         |
| **Pandas**     | Load and manipulate CSV data                         |
| **Scikit-learn** | Train and predict with RandomForestRegressor      |
| **Joblib**     | Save/load the ML model (`model.pkl`)                 |
| **OpenAI** *(optional)* | Chatbot feature (currently disabled)       |

---

## 📄 Expected CSV Format

Your uploaded `.csv` file should include the following columns:

| Column Name     | Description                                |
|----------------|--------------------------------------------|
| `Campaign`      | Name of the marketing campaign             |
| `Impressions`   | Number of times the ad was shown           |
| `Clicks`        | Number of times the ad was clicked         |
| `Spend (₹)`     | Total ad spend in INR                      |
| `Conversions`   | Number of successful outcomes (leads/sales)|

### 🧪 Example CSV Preview:

| Campaign   | Impressions | Clicks | Spend (₹) | Conversions |
|------------|-------------|--------|-----------|-------------|
| Campaign A | 10000       | 250    | 5000      | 30          |
| Campaign B | 8000        | 160    | 4000      | 18          |
| Campaign C | 15000       | 300    | 7000      | 25          |

---

## 📦 Installation

```bash
git clone https://github.com/mishraansh07/Marketmind.ai.git
cd Marketmind.ai
pip install -r requirements.txt
streamlit run main.py
