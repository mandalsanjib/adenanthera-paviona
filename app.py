import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Hide TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Plant Growth Stage AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    h1, h2, h3 { color: #f8fafc !important; }
    .prediction-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #475569;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 16px;
    }
    .metric-card {
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.25);
    }
    .stButton > button {
        background: linear-gradient(90deg, #14b8a6, #0d9488) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #0d9488, #0f766e) !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.4);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
        border-right: 1px solid #334155;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================== LOAD RESOURCES ======================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("best_model.keras")
    with open("class_names.json") as f:
        class_names = json.load(f)
    with open("stage_details.json") as f:
        stage_details = json.load(f)
    return model, class_names, stage_details

try:
    model, class_names, STAGE_DETAILS = load_resources()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")

IMG_SIZE = (224, 224)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### 🌱 Plant Stage AI")
    st.markdown("---")
    st.markdown("**Detects 9 growth stages:**")
    for name in class_names:
        st.caption(f"• {name.title()}")
    st.markdown("---")
    st.info("Upload a clear photo of a seed or plant for best results.")
    st.caption(f"Updated: {datetime.now().strftime('%b %Y')}")

# ====================== MAIN UI ======================
st.markdown("<h1 style='text-align:center;'>🌱 Plant Growth Stage Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:17px;'>Upload an image → Get instant stage prediction + care advice</p>", unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader("Upload plant / seed image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1.1, 1])
    
    with col1:
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### Analysis")
        if st.button("🔍 Analyze Stage", use_container_width=True):
            with st.spinner("AI is analyzing..."):
                # Preprocess
                img = image.resize(IMG_SIZE)
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                preds = model.predict(img_array, verbose=0)[0]
                pred_idx = np.argmax(preds)
                confidence = float(preds[pred_idx])
                pred_class = class_names[pred_idx]
                details = STAGE_DETAILS.get(pred_class, {
                    "name": pred_class.title(),
                    "description": "No detailed description available.",
                    "key_features": "—",
                    "care_tips": "—"
                })
                
                # Result card
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="margin:0; color:#2dd4bf;">{details.get('name', pred_class.title())}</h2>
                    <p style="color:#94a3b8; margin:4px 0 0 0;">Detected Growth Stage</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence
                conf_color = "#10b981" if confidence > 0.70 else "#f59e0b" if confidence > 0.50 else "#ef4444"
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(145deg, {conf_color}, {conf_color}cc);">
                    <h3 style="margin:0; color:white;">Confidence</h3>
                    <h1 style="margin:6px 0; color:white;">{confidence*100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📋 Description", expanded=True):
                    st.write(details.get("description", "—"))
                with st.expander("🔍 Key Features"):
                    st.write(details.get("key_features", "—"))
                with st.expander("💧 Care Tips"):
                    st.write(details.get("care_tips", "—"))
                if "duration" in details:
                    with st.expander("⏱ Typical Duration"):
                        st.write(details["duration"])
    
    # ===== REAL-TIME CHART =====
    if st.session_state.get("analyzed", False) or st.button:  # simple trigger
        pass

    # We re-run prediction only when button is clicked (already done above)
    # For the chart we need the preds variable in scope → better structure:

# Better structure: move prediction outside and store in session
if uploaded_file is not None and model_loaded:
    if "last_preds" not in st.session_state:
        st.session_state.last_preds = None
        st.session_state.last_class = None
        st.session_state.last_details = None
        st.session_state.last_conf = None

    if st.button("🔍 Analyze Stage", key="analyze_btn", use_container_width=True) or st.session_state.last_preds is not None:
        if st.session_state.last_preds is None or st.button:  # force re-predict only on click
            img = image.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds = model.predict(img_array, verbose=0)[0]
            
            pred_idx = np.argmax(preds)
            st.session_state.last_preds = preds
            st.session_state.last_class = class_names[pred_idx]
            st.session_state.last_conf = float(preds[pred_idx])
            st.session_state.last_details = STAGE_DETAILS.get(class_names[pred_idx], {})
        
        preds = st.session_state.last_preds
        pred_class = st.session_state.last_class
        confidence = st.session_state.last_conf
        details = st.session_state.last_details
        
        # Chart section
        st.markdown("---")
        st.markdown("### 📊 Confidence Across All Stages")
        
        df = pd.DataFrame({
            "Stage": [c.title() for c in class_names],
            "Confidence": preds * 100
        }).sort_values("Confidence", ascending=True)
        
        colors = ['#14b8a6' if c.lower() == pred_class.lower() else '#475569' for c in df["Stage"]]
        
        fig = go.Figure(go.Bar(
            y=df["Stage"],
            x=df["Confidence"],
            orientation='h',
            marker_color=colors,
            text=[f"{v:.1f}%" for v in df["Confidence"]],
            textposition='outside',
            textfont=dict(color='#e2e8f0', size=12),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>"
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            xaxis=dict(title="Confidence (%)", gridcolor='#334155', range=[0, 110]),
            yaxis=dict(color='#e2e8f0'),
            height=460,
            margin=dict(l=10, r=40, t=20, b=40),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top 3
        st.markdown("#### Top 3 Predictions")
        top3 = np.argsort(preds)[-3:][::-1]
        tcols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for i, idx in enumerate(top3):
            with tcols[i]:
                bg = "linear-gradient(145deg, #0f766e, #14b8a6)" if i == 0 else "linear-gradient(145deg, #1e293b, #334155)"
                st.markdown(f"""
                <div style="background:{bg}; border-radius:12px; padding:16px; text-align:center; border:1px solid #475569;">
                    <h4 style="margin:0; color:white;">{medals[i]} {class_names[idx].title()}</h4>
                    <h2 style="margin:8px 0; color:white;">{preds[idx]*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:70px 20px; background:rgba(30,41,59,0.5); 
                border-radius:16px; border:2px dashed #475569; margin-top:40px;">
        <h2 style="color:#94a3b8;">📷 Upload an image to begin</h2>
        <p style="color:#64748b;">JPG • PNG • WEBP supported</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b;'>Plant Growth Stage AI • Powered by TensorFlow + Streamlit</p>", unsafe_allow_html=True)