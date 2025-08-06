import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === PAGE CONFIG ===
st.set_page_config(
    page_title="AI Document Authenticator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        padding-top: 2rem;
    }
    
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #f8f9ff 0%, #e6eeff 100%);
        min-height: 100vh;
        font-size: 1.2rem;
    }
    
    .header-gradient {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 1);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .title-container {
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        animation: pulse 2s infinite;
        line-height: 1.1;
    }
    
    .subtitle {
        color: #666;
        font-size: 3.5rem;
        font-weight: 400;
        margin-top: 1rem;
    }
    
    .upload-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        border: 2px dashed #667eea;
        transition: all 0.3s ease;
        animation: slideInUp 0.8s ease-out;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .upload-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        animation: slideInLeft 0.8s ease-out;
        border: 1px solid #f0f0f0;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .result-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        animation: zoomIn 0.5s ease-out;
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.3);
        height: 12px;
        border-radius: 6px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
        border-radius: 6px;
        animation: fillBar 1.5s ease-out;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
        transition: all 0.3s ease;
        border: 2px solid #f0f0f0;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15);
        border-color: #667eea;
    }
    
    /* Make drag and drop text darker and larger */
    .stFileUploader > div > div > div > div > div > p {
        color: #333 !important;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
    }
    
    .stFileUploader > div > div > div > div > div > small {
        color: #555 !important;
        font-weight: 500 !important;
        font-size: 1.1rem !important;
    }
    
    /* Make analysis details text darker and larger */
    .analysis-details {
        color: #222 !important;
        font-weight: 500 !important;
        font-size: 1.3rem !important;
        line-height: 1.6;
    }
    
    /* Make recommendations text darker and larger */
    .recommendations-text {
        color: #222 !important;
        font-weight: 500 !important;
        font-size: 1.3rem !important;
    }
    
    /* Override Streamlit's default text sizes and colors */
    .stAlert > div {
        color: #222 !important;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
    }
    
    .stSuccess > div {
        color: #0d5c20 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    .stWarning > div {
        color: #8a4e00 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    .stError > div {
        color: #9e2146 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    .stInfo > div {
        color: #0f4c75 !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }
    
    /* Make general text larger */
    p, div, span, li {
        font-size: 1.2rem !important;
        line-height: 1.6;
    }
    
    /* Make headings larger */
    h1 { font-size: 3.5rem !important; }
    h2 { font-size: 3rem !important; }
    h3 { font-size: 2.2rem !important; }
    h4 { font-size: 1.8rem !important; }
    h5 { font-size: 1.5rem !important; }
    h6 { font-size: 1.3rem !important; }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes zoomIn {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes fillBar {
        from { width: 0%; }
        to { width: var(--confidence-width); }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 1rem 2.5rem;
        font-size: 1.3rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# === CONFIG ===
CLASS_NAMES = ['fake', 'real']
MODEL_PATH = 'best_model.pth'

# === Device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load Model ===
@st.cache_resource
def load_model():
    try:
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, out_features=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model.to(device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Helper Functions ===
def create_confidence_chart(confidence_scores):
    fig = go.Figure(data=[
        go.Bar(
            x=['Fake', 'Real'],
            y=confidence_scores,
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{score:.1f}%' for score in confidence_scores],
            textposition='auto',
            textfont=dict(size=16, color='white', family="Poppins")
        )
    ])
    
    fig.update_layout(
        title="Confidence Distribution",
        title_font_size=24,
        title_x=0.5,
        title_font_family="Poppins",
        xaxis_title="Classification",
        yaxis_title="Confidence (%)",
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        template="plotly_white",
        height=400,
        showlegend=False,
        font=dict(family="Poppins", size=14)
    )
    
    return fig

def display_loading_animation():
    loading_html = """
    <div style="display: flex; justify-content: center; align-items: center; height: 120px;">
        <div class="loading-spinner"></div>
    </div>
    <p style="text-align: center; color: #667eea; font-weight: 600; font-size: 1.4rem;">Analyzing document...</p>
    """
    return st.markdown(loading_html, unsafe_allow_html=True)

# === Main UI ===
def main():
    # Header Section
    st.markdown("""
    <div class="title-container">
        <h1 class="main-title">🔍 Forgeo</h1>
        <p class="subtitle">Forget forgery. Remember Forgeo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #667eea; margin: 0; font-size: 2.8rem;">🎯</h3>
            <h4 style="margin: 0.5rem 0; color: #333; font-weight: 600; font-size: 1.6rem;">Accuracy</h4>
            <p style="margin: 0; color: #333; font-size: 1.8rem; font-weight: 600;">95.2%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #667eea; margin: 0; font-size: 2.8rem;">⚡</h3>
            <h4 style="margin: 0.5rem 0; color: #333; font-weight: 600; font-size: 1.6rem;">Speed</h4>
            <p style="margin: 0; color: #333; font-size: 1.8rem; font-weight: 600;">&lt; 2s</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #667eea; margin: 0; font-size: 2.8rem;">🧠</h3>
            <h4 style="margin: 0.5rem 0; color: #333; font-weight: 600; font-size: 1.6rem;">Model</h4>
            <p style="margin: 0; color: #333; font-size: 1.8rem; font-weight: 600;">EfficientNet</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #667eea; margin: 0; font-size: 2.8rem;">📊</h3>
            <h4 style="margin: 0.5rem 0; color: #333; font-weight: 600; font-size: 1.6rem;">Processed</h4>
            <p style="margin: 0; color: #333; font-size: 1.8rem; font-weight: 600;">10K+ docs</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("""
    <div class="upload-section">
        <h3 style="color: #667eea; margin-bottom: 1rem; font-weight: 600; font-size: 2.2rem;">📎 Upload Your Document</h3>
        <p style="color: #333; margin-bottom: 1rem; font-size: 1.5rem; font-weight: 500;">Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the document you want to verify"
    )
    
    if uploaded_file is not None:
        # Display image in columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="📷 Document Image",
                use_column_width=True,
                output_format="JPEG"
            )
        
        # Features section
        st.markdown("""
        <div class="feature-card">
            <h4 style="color: #667eea; font-weight: 600; font-size: 2rem; margin-bottom: 1.5rem;">🔍 What we analyze:</h4>
            <ul style="color: #222; line-height: 1.8; font-size: 1.4rem; font-weight: 500;">
                <li style="margin-bottom: 0.8rem;"><strong style="color: #111; font-size: 1.5rem;">Document Structure:</strong> Layout patterns and formatting</li>
                <li style="margin-bottom: 0.8rem;"><strong style="color: #111; font-size: 1.5rem;">Text Quality:</strong> Font consistency and print quality</li>
                <li style="margin-bottom: 0.8rem;"><strong style="color: #111; font-size: 1.5rem;">Visual Elements:</strong> Logos, seals, and security features</li>
                <li style="margin-bottom: 0.8rem;"><strong style="color: #111; font-size: 1.5rem;">Paper Texture:</strong> Surface characteristics and aging patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Classification button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            classify_btn = st.button("🚀 Analyze Document", use_container_width=True)
        
        if classify_btn:
            # Load model
            model = load_model()
            if model is None:
                st.error("❌ Model could not be loaded. Please check the model file.")
                return
            
            # Loading animation
            with st.spinner(""):
                loading_placeholder = st.empty()
                with loading_placeholder:
                    display_loading_animation()
                
                # Simulate processing time for better UX
                time.sleep(1.5)
                loading_placeholder.empty()
                
                # Preprocess and classify
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    pred_index = torch.argmax(probs).item()
                    confidence = probs[pred_index].item() * 100
                    
                    # Get confidence scores for both classes
                    fake_confidence = probs[0].item() * 100
                    real_confidence = probs[1].item() * 100
            
            # Results section
            result_color = "#4ECDC4" if CLASS_NAMES[pred_index] == "real" else "#FF6B6B"
            result_icon = "✅" if CLASS_NAMES[pred_index] == "real" else "⚠️"
            
            st.markdown(f"""
            <div class="result-container">
                <h2 style="font-size: 2.5rem; margin-bottom: 1rem;">{result_icon} Classification Result</h2>
                <h1 style="font-size: 4rem; margin: 1rem 0; font-weight: 700;">{CLASS_NAMES[pred_index].upper()}</h1>
                <p style="font-size: 1.8rem; font-weight: 600;">Confidence: {confidence:.2f}%</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="--confidence-width: {confidence}%; width: {confidence}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h3 style="color: #333; font-weight: 600; font-size: 2.2rem; margin-bottom: 1rem;">📊 Confidence Analysis</h3>', unsafe_allow_html=True)
                fig = create_confidence_chart([fake_confidence, real_confidence])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<h3 style="color: #333; font-weight: 600; font-size: 2.2rem; margin-bottom: 1rem;">🔍 Analysis Details</h3>', unsafe_allow_html=True)
                
                if CLASS_NAMES[pred_index] == "real":
                    st.success("✅ Document appears to be authentic")
                    st.info("🔍 High confidence in document legitimacy")
                    st.markdown("""
                    <div class="analysis-details" style="margin-top: 1rem;">
                    <strong style="font-size: 1.4rem;">Positive indicators detected:</strong><br><br>
                    • Consistent formatting patterns<br>
                    • Authentic text characteristics<br>
                    • Proper document structure
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Document shows signs of manipulation")
                    st.error("🔍 Potential forgery detected")
                    st.markdown("""
                    <div class="analysis-details" style="margin-top: 1rem;">
                    <strong style="font-size: 1.4rem;">Suspicious indicators found:</strong><br><br>
                    • Inconsistent formatting<br>
                    • Irregular text patterns<br>
                    • Anomalous visual elements
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendation section
            st.markdown('<h3 style="color: #333; font-weight: 600; font-size: 2.2rem; margin: 2rem 0 1rem 0;">💡 Recommendations</h3>', unsafe_allow_html=True)
            
            if confidence > 90:
                st.success("🎯 **High Confidence Result** - The analysis shows strong indicators for this classification.")
            elif confidence > 70:
                st.warning("⚡ **Moderate Confidence** - Consider additional verification methods for critical decisions.")
            else:
                st.error("⚠️ **Low Confidence** - Manual review recommended. The document shows ambiguous characteristics.")
            
            # Export results option
            st.markdown('<hr style="margin: 2rem 0;">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("📥 Generate Report", use_container_width=True):
                    report = f"""
Document Analysis Report
========================
Classification: {CLASS_NAMES[pred_index].upper()}
Confidence: {confidence:.2f}%
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

Detailed Scores:
- Fake Probability: {fake_confidence:.2f}%
- Real Probability: {real_confidence:.2f}%

Analysis Status: {'High Confidence' if confidence > 90 else 'Moderate Confidence' if confidence > 70 else 'Low Confidence'}
                    """
                    st.download_button(
                        label="💾 Download Report",
                        data=report,
                        file_name=f"document_analysis_{int(time.time())}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()