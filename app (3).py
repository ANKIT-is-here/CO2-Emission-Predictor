import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fpdf import FPDF
import io
import tempfile
import os
import random
import time

@st.cache_resource
def get_model(model_name):
    if model_name == "Random Forest":
        return load("optimized_random_forest_model.joblib")
    else:
        return load("optimized_xgboost_model.joblib")

st.set_page_config(page_title="Ship Emission Predictor", page_icon="🌍", layout="wide")
st.markdown("""
    <style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; font-size: 18px; font-weight: 600; }
    .stApp { background-image: linear-gradient(to bottom right, #e6ffe6, #ccffcc); }
    .navbar { position: sticky; top: 0; z-index: 999; background-color: #007744; padding: 16px; color: white; font-size: 28px; font-weight: bold; text-align: center; border-radius: 0 0 12px 12px; letter-spacing: 1px; box-shadow: 0px 2px 8px rgba(0,0,0,0.2);}
    .vnit-logo { position: absolute; top: 8px; left: 20px; height: 60px; }
    .flashcards-container { display: flex; width: 100%; justify-content: flex-start; gap: 18px; margin-bottom: 18px; }
    .flashcard { flex: 0 0 32%; background: linear-gradient(120deg, #e6ffe6 60%, #b2f7b8 100%); border-radius: 18px; box-shadow: 0 4px 24px rgba(0,119,68,0.10); min-height: 90px; padding: 20px 14px; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; font-weight: 700; color: #007744; text-align: center; border: 2px solid #b2f7b8; transition: box-shadow 0.3s, background 0.3s, color 0.3s;}
    .flashcard.active { background: linear-gradient(120deg, #c7ffd8 60%, #00b378 100%); color: #fff; border: 2px solid #00b378; animation: slideRight 0.8s cubic-bezier(.4,1.5,.5,1) 1;}
    @keyframes slideRight { 0% { transform: translateX(-60%); opacity: 0;} 40% { opacity: 1;} 100% { transform: translateX(0); opacity: 1;} }
    .share-btn-row { display: flex; gap: 18px; margin-top: 10px; margin-bottom: 30px; }
    .share-btn { display: inline-flex; align-items: center; justify-content: center; padding: 8px 22px; border-radius: 8px; font-size: 1.1rem; font-weight: 700; border: none; text-decoration: none !important; transition: background 0.2s, color 0.2s; box-shadow: 0 2px 6px rgba(0,119,68,0.07); margin-right: 6px; margin-bottom: 6px;}
    .share-btn.linkedin { background: #0174b2; color: #fff !important;}
    .share-btn.linkedin:hover { background: #005983; color: #fff !important;}
    .share-btn.gmail { background: #ea4335; color: #fff !important;}
    .share-btn.gmail:hover { background: #b23121; color: #fff !important;}
    .share-btn svg { margin-right: 8px; vertical-align: middle;}
    </style>
    <div class='navbar'>
        <img src='vnit_logo.png' class='vnit-logo'>
        🌱 <span style='color:white;'>Ship CO2 Emission Predictor</span>
    </div>
""", unsafe_allow_html=True)

# --- Quotes/flashcards section (greenish, 3s flip) ---
quotes = [
    "🌱 The Earth is what we all have in common. — Wendell Berry",
    "🌍 Small acts, when multiplied by millions, can transform the world.",
    "💧 Save water, save life, save the planet.",
    "🌳 Plant a tree today for a greener tomorrow.",
    "♻️ Reduce, reuse, recycle for a better world.",
    "🌞 The greatest threat to our planet is the belief that someone else will save it.",
    "🌿 Every action counts in the fight against climate change.",
    "🚢 Greener shipping means a cleaner future.",
    "🌲 Nature does not hurry, yet everything is accomplished. — Lao Tzu"
]
FLASHCARD_INTERVAL = 3
if "flashcard_state" not in st.session_state:
    st.session_state.flashcard_state = [random.choice(quotes) for _ in range(3)]
    st.session_state.active_idx = random.randint(0,2)
    st.session_state.last_update = time.time()
now = time.time()
if now - st.session_state.last_update >= FLASHCARD_INTERVAL:
    idx = random.randint(0,2)
    st.session_state.flashcard_state[idx] = random.choice(quotes)
    st.session_state.active_idx = idx
    st.session_state.last_update = now
def flashcard_html(quotes, active_idx):
    html = '<div class="flashcards-container">'
    for i, q in enumerate(quotes):
        card_class = "flashcard active" if i == active_idx else "flashcard"
        html += f'<div class="{card_class}">{q if i == active_idx else "&nbsp;"}</div>'
    html += '</div>'
    return html
st.markdown(flashcard_html(st.session_state.flashcard_state, st.session_state.active_idx), unsafe_allow_html=True)

st.sidebar.title("🔧 Navigation")
page = st.sidebar.radio("Go to", ["📊 Predict Emissions", "📁 Upload CSV", "📜 Policy Suggestions"])
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "XGBoost"])

if page == "📊 Predict Emissions":
    st.subheader("Enter Ship Parameters Manually")
    distance = st.slider("Distance (in kilometers)", 100, 20000, 5000)
    engine_efficiency = st.slider("Engine Efficiency", 0.0, 1.0, 0.85)
    fuel_type = st.selectbox("Fuel Type", ["HFO", "Diesel"])
    emission_efficiency = st.slider("Emission Efficiency (kg CO₂/litre)", 1.00, 5.00, 3.00, step=0.01)
    st.caption("📌 Emission efficiency is user-set (kg CO₂/litre) | Distance in km | Engine efficiency between 0.0 and 1.0")
    st.metric(label="Emission Efficiency", value=f"{emission_efficiency:.2f} kg CO₂/litre")
    features = np.array([[distance, engine_efficiency, emission_efficiency]])
    if st.button("🌍 Predict Emission"):
        model = get_model(model_choice)
        result = model.predict(features)[0]
        trees_required = result / 21
        st.success(f"🌿 Predicted CO2 Emission: {result:.2f} kg")
        st.markdown(f"""
<div class="did-you-know-box" style="
    background-color: #f0fff0;
    border-left: 6px solid #2e8b57;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-size: 17px;
    box-shadow: 2px 2px 10px rgba(0, 100, 0, 0.1);
">
    🌳 <strong>Did you know?</strong><br><br>
    To absorb <strong>{result:.2f} kg</strong> of CO₂ emitted from this journey in one year,<br>
    we would need to plant approximately <strong>{trees_required:.0f} trees</strong> 🌱.
    <br><br>
    Let’s strive for greener voyages and a cleaner planet! 🌍
</div>
""", unsafe_allow_html=True)
        st.markdown("### 🔍 Feature Importance (Random Forest)")
        feature_importances = pd.DataFrame({
            'Feature': ['distance', 'emission_efficiency', 'engine_efficiency'],
            'Importance': [0.944851, 0.030913, 0.024236]
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        st.dataframe(feature_importances.style.bar(subset=['Importance'], color='#4CAF50'))
        st.markdown("### 📈 Trade-off Visualization Based on Your Input")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        dist_range = np.linspace(distance * 0.8, distance * 1.2, 30)
        eff_range = np.linspace(max(0.1, engine_efficiency - 0.2), min(1.0, engine_efficiency + 0.2), 30)
        dist_grid, eff_grid = np.meshgrid(dist_range, eff_range)
        inputs = np.array([dist_grid.ravel(), eff_grid.ravel(), [emission_efficiency]*len(dist_grid.ravel())]).T
        preds = model.predict(inputs).reshape(dist_grid.shape)
        surf = ax.plot_surface(dist_grid, eff_grid, preds, cmap='viridis')
        ax.set_title("3D Trade-off: Distance vs Efficiency vs Emissions")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Engine Efficiency")
        ax.set_zlabel("Predicted CO2 Emission (kg)")
        fig.colorbar(surf, shrink=0.5, aspect=10)
        st.pyplot(fig)
        st.markdown("### 📄 Generate Final Report")
        st.markdown("### 📋 Personalize Your Report")
        uploaded_logo = st.file_uploader("Upload your logo for the report (optional)", type=["png", "jpg", "jpeg"])
        user_notes = st.text_area("Add custom notes to your report (optional)")
        section_options = {
            "Input Parameters": True,
            "3D Emission Visualization": True,
            "Feature Importance": st.checkbox("Include Feature Importance Table", value=True),
            "Thank You Message": True
        }
        class PDF(FPDF):
            def header(self):
                if uploaded_logo is not None:
                    logo_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    uploaded_logo.seek(0)
                    with open(logo_path, "wb") as f:
                        f.write(uploaded_logo.read())
                    self.image(logo_path, x=10, y=8, w=25)
                self.set_fill_color(0, 119, 68)
                self.rect(0, 0, 210, 18, 'F')
                self.set_font("DejaVu", 'B', 18)
                self.set_text_color(255,255,255)
                self.cell(0, 12, "🌱 Ship CO2 Emission Prediction Report", ln=True, align="C")
                self.ln(5)
            def footer(self):
                self.set_y(-15)
                self.set_font("DejaVu", "", 10)
                self.set_text_color(120,120,120)
                self.cell(0, 10, f"Page {self.page_no()}  |  Powered by VNIT Nagpur", 0, 0, "C")
        pdf = PDF()
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        pdf.add_page()
        pdf.set_font("DejaVu", "", 10)
        pdf.cell(0, 0, "")  # Dummy use to register font
        if section_options["Input Parameters"]:
            pdf.set_font("DejaVu", 'B', 14)
            pdf.set_text_color(0,119,68)
            pdf.cell(0, 10, "Input Parameters", ln=True)
            pdf.set_text_color(0,0,0)
            pdf.set_font("DejaVu", '', 12)
            pdf.cell(0, 8, f"Distance: {distance} km", ln=True)
            pdf.cell(0, 8, f"Engine Efficiency: {engine_efficiency:.2f}", ln=True)
            pdf.cell(0, 8, f"Fuel Type: {fuel_type}", ln=True)
            pdf.cell(0, 8, f"Emission Efficiency: {emission_efficiency:.2f} kg CO₂/litre", ln=True)
            pdf.cell(0, 8, f"Predicted CO2 Emission: {result:.2f} kg", ln=True)
            pdf.ln(5)
        if user_notes:
            pdf.set_font("DejaVu", 'B', 12)
            pdf.set_text_color(0,119,68)
            pdf.cell(0, 10, "Your Notes", ln=True)
            pdf.set_font("DejaVu", '', 11)
            pdf.set_text_color(0,0,0)
            pdf.multi_cell(0, 8, user_notes)
            pdf.ln(2)
        if section_options["3D Emission Visualization"]:
            pdf.set_font("DejaVu", 'B', 14)
            pdf.set_text_color(0,119,68)
            pdf.cell(0, 10, "3D Emission Prediction Visualization", ln=True)
            tmp_graph = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            fig.savefig(tmp_graph.name, bbox_inches='tight')
            pdf.image(tmp_graph.name, x=10, w=190)
            pdf.ln(5)
        if section_options["Feature Importance"]:
            pdf.set_font("DejaVu", 'B', 12)
            pdf.set_text_color(0,119,68)
            pdf.cell(0, 10, "Feature Importance (Random Forest)", ln=True)
            pdf.set_font("DejaVu", '', 11)
            pdf.set_text_color(0,0,0)
            pdf.cell(0, 8, "distance: 0.9449", ln=True)
            pdf.cell(0, 8, "emission_efficiency: 0.0309", ln=True)
            pdf.cell(0, 8, "engine_efficiency: 0.0242", ln=True)
            pdf.ln(3)
        if section_options["Thank You Message"]:
            pdf.set_fill_color(204,255,204)
            pdf.set_text_color(0,119,68)
            pdf.set_font("DejaVu", 'B', 12)
            pdf.cell(0, 10, "Thank you for using Ship CO2 Emission Predictor!", ln=True, fill=True)
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button(
            label="📥 Download Personalized Report (PDF)",
            data=pdf_buffer,
            file_name="Ship_Emission_Report.pdf",
            mime="application/pdf"
        )
        share_url = "https://your-app-url.com"
        share_title = "Ship CO2 Emission Prediction"
        share_summary = f"I just predicted my ship's CO₂ emissions with this awesome app! Result: {result:.2f} kg CO₂. Try it yourself!"
        linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url={share_url}"
        gmail_url = f"https://mail.google.com/mail/?view=cm&fs=1&to=&su={share_title}&body={share_summary}%0A{share_url}"
        st.markdown(f"""
        <div class="share-btn-row">
            <a href="{linkedin_url}" target="_blank" class="share-btn linkedin">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="white" viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.761 0 5-2.239 5-5v-14c0-2.761-2.239-5-5-5zm-11 19h-3v-10h3v10zm-1.5-11.268c-.966 0-1.75-.78-1.75-1.732s.784-1.732 1.75-1.732 1.75.78 1.75 1.732-.784 1.732-1.75 1.732zm13.5 11.268h-3v-5.604c0-1.337-.025-3.063-1.868-3.063-1.868 0-2.154 1.459-2.154 2.967v5.7h-3v-10h2.881v1.367h.041c.401-.761 1.381-1.561 2.841-1.561 3.04 0 3.601 2.002 3.601 4.604v5.59z"/></svg>
                LinkedIn
            </a>
            <a href="{gmail_url}" target="_blank" class="share-btn gmail">
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="white" viewBox="0 0 24 24"><path d="M12 13.065l-11.715-7.065h23.43l-11.715 7.065zm11.715-8.065c.158-.09.285-.232.285-.414v-2.586c0-.276-.224-.5-.5-.5h-23c-.276 0-.5.224-.5.5v2.586c0 .182.127.324.285.414l11.715 7.065 11.715-7.065zm-23.715 2.065v13c0 .276.224.5.5.5h23c.276 0 .5-.224.5-.5v-13l-12 7.239-12-7.239z"/></svg>
                Gmail
            </a>
        </div>
        """, unsafe_allow_html=True)
        # --- Comparison Section ---
        st.markdown("### 🟩 Emission Comparison: Varying Engine Efficiency")
        st.write(f"**Distance fixed at:** `{distance}` km, **Emission Efficiency fixed at:** `{emission_efficiency:.2f}`")
        comp_results = []
        for eff in np.linspace(0.5, 1.0, 11):
            test_feat = np.array([[distance, eff, emission_efficiency]])
            pred = model.predict(test_feat)[0]
            trees = pred / 21
            comp_results.append({"Engine Efficiency": eff, "Predicted CO₂ Emission (kg)": pred, "Trees Needed": trees})
        comp_df = pd.DataFrame(comp_results)
        st.dataframe(comp_df.style.background_gradient(subset=["Predicted CO₂ Emission (kg)"], cmap="Greens_r"))
        min_emission = comp_df["Predicted CO₂ Emission (kg)"].min()
        max_emission = comp_df["Predicted CO₂ Emission (kg)"].max()
        max_trees = comp_df["Trees Needed"].max()
        min_trees = comp_df["Trees Needed"].min()
        st.markdown(f"""
- **Lowest emission**: {min_emission:.2f} kg CO₂ at engine efficiency {comp_df.iloc[comp_df['Predicted CO₂ Emission (kg)'].idxmin()]['Engine Efficiency']:.2f}
- **Highest emission**: {max_emission:.2f} kg CO₂ at engine efficiency {comp_df.iloc[comp_df['Predicted CO₂ Emission (kg)'].idxmax()]['Engine Efficiency']:.2f}
- **Trees saved** by improving efficiency from 0.5 to 1.0: {max_trees - min_trees:.0f}
""")
        st.markdown("### 🤖 AI Tips for Maximum Efficiency")
        st.markdown("""
- **Slow Steaming:** Reduce speed where possible. Even a 10% reduction in speed can cut emissions by up to 20–30%.
- **Sail at Optimum Speed:** For displacement ships, aim for about two-thirds of hull speed for best fuel efficiency.
- **Route Optimization:** Use shortest, straight-line routes and avoid unnecessary deviations. Use weather routing to avoid strong headwinds and currents.
- **Trim Optimization:** Balance ship trim to reduce drag and improve efficiency.
- **Autopilot Optimization:** Minimize rudder movements and maintain a steady, straight course.
- **Hull & Propeller Maintenance:** Keep hull and propeller clean to reduce drag and improve fuel efficiency.
- **Regular Engine Maintenance:** Maintain engine and components for optimal performance and lower emissions.
""")
        st.info("**Pro tip:** Combining slow steaming, route optimization, and regular maintenance can reduce fuel and emission by 15–30% per voyage!")

elif page == "📁 Upload CSV":
    st.subheader("📤 Upload CSV for Emission Prediction")
    st.markdown(
        '<div style="background-color:#fffbe6;border-left:5px solid #ffa500;padding:12px 18px;border-radius:8px;margin-bottom:18px;font-size:16px;">'
        '<b>Note:</b> The uploaded file should contain only the columns: '
        '<span style="color:#007744;">"distance"</span>, '
        '<span style="color:#007744;">"engine efficiency"</span>, and '
        '<span style="color:#007744;">"emission efficiency"</span>.'
        '</div>',
        unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload a CSV file with appropriate features", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(df)
        model = get_model(model_choice)
        predictions = model.predict(df)
        df["Predicted CO2 Emission (kg)"] = predictions
        st.success("✅ Emissions predicted for uploaded data!")
        st.dataframe(df)
        st.bar_chart(df["Predicted CO2 Emission (kg)"])
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111, projection='3d')
        dist_range = np.linspace(df['distance'].min(), df['distance'].max(), 30)
        eff_range = np.linspace(df['engine efficiency'].min(), df['engine efficiency'].max(), 30)
        dist_grid, eff_grid = np.meshgrid(dist_range, eff_range)
        emission_eff = df['emission efficiency'].mean()
        inputs = np.array([dist_grid.ravel(), eff_grid.ravel(), [emission_eff]*len(dist_grid.ravel())]).T
        preds = model.predict(inputs).reshape(dist_grid.shape)
        surf = ax.plot_surface(dist_grid, eff_grid, preds, cmap='viridis')
        ax.set_title("3D Trade-off: Distance vs Efficiency vs Emissions")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Engine Efficiency")
        ax.set_zlabel("Predicted CO2 Emission (kg)")
        plt.tight_layout()
        st.pyplot(fig)
        class PDF(FPDF):
            def header(self):
                self.set_fill_color(0, 119, 68)
                self.rect(0, 0, 210, 18, 'F')
                self.set_font("DejaVu", 'B', 18)
                self.set_text_color(255,255,255)
                self.cell(0, 12, "🌱 Ship CO2 Emission Prediction Report", ln=True, align="C")
                self.ln(5)
            def footer(self):
                self.set_y(-15)
                self.set_font("DejaVu", "", 10)
                self.set_text_color(120,120,120)
                self.cell(0, 10, f"Page {self.page_no()}  |  Powered by VNIT Nagpur", 0, 0, "C")
        pdf = PDF()
        pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
        pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
        pdf.add_page()
        pdf.set_font("DejaVu", 'B', 14)
        pdf.set_text_color(0,119,68)
        pdf.cell(0, 10, "Summary of Uploaded Data", ln=True)
        pdf.set_text_color(0,0,0)
        pdf.set_font("DejaVu", '', 11)
        for col in df.columns:
            pdf.cell(0, 8, f"{col}: {df[col].iloc[0]}", ln=True)
        pdf.ln(3)
        pdf.set_font("DejaVu", 'B', 12)
        pdf.set_text_color(0,119,68)
        pdf.cell(0, 10, "3D Emission Prediction Visualization", ln=True)
        tmp_graph = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(tmp_graph.name, bbox_inches='tight')
        pdf.image(tmp_graph.name, x=10, w=190)
        pdf.ln(5)
        pdf.set_font("DejaVu", '', 11)
        pdf.set_text_color(0,0,0)
        pdf.multi_cell(0, 8, "This report provides a professional summary of your uploaded ship data and the predicted CO2 emissions, including a 3D visualization of the relationship between distance, engine efficiency, and emission efficiency. For more details, visit our website or contact the VNIT Nagpur team.")
        pdf.ln(5)
        pdf.set_fill_color(204,255,204)
        pdf.set_text_color(0,119,68)
        pdf.set_font("DejaVu", 'B', 12)
        pdf.cell(0, 10, "Thank you for using Ship CO2 Emission Predictor!", ln=True, fill=True)
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button(
            label="📥 Download Professional Report (PDF)",
            data=pdf_buffer,
            file_name="Ship_Emission_Report.pdf",
            mime="application/pdf"
        )
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📩 Download CSV with Predictions",
            data=csv,
            file_name="predicted_emissions.csv",
            mime="text/csv"
        )

elif page == "📜 Policy Suggestions":
    st.subheader("📜 Suggest Eco-Friendly Maritime Policies")
    st.markdown("""
    ✅ Here are some ideas to get you started:
    - Encourage the use of **clean fuels** like LNG or biofuels
    - Enforce **emission control areas (ECAs)** near coastal cities
    - Incentivize **AI-powered route optimization**
    - Reward **low-emission ship designs**
    """)
    suggestion = st.text_area("💡 Enter your policy suggestion:", max_chars=500)
    name = st.text_input("✍️ Your Name (Optional):")
    if st.button("📤 Submit Suggestion"):
        if suggestion.strip():
            st.success("✅ Thank you for your valuable input! Your idea has been recorded.")
        else:
            st.warning("⚠️ Suggestion cannot be empty.")
    st.markdown("---")
    st.markdown("🌍 Every idea counts. Together, we can build a greener maritime future.")

# --- Footer ---
st.markdown("""
    <hr style="margin-top:40px; margin-bottom:10px; border: none; border-top: 2px solid #007744;">
    <div style="text-align:center; color:#007744; font-weight:700; font-size:1.1rem;">
        VNIT Nagpur × IIT Kharagpur
    </div>
""", unsafe_allow_html=True)
