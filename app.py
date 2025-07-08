import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import time

# Set page config
st.set_page_config(
    page_title="AgriScan Plant Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a2f1a 0%, #3d6b3d 100%);
        color: #fff;
    }
    h1, h2, h3 {
        color: #d4f0d4 !important;
    }
    .block-container {
        background: rgba(0, 20, 0, 0.85);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(to bottom, #4CAF50, #2E7D32) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        padding: 10px 24px !important;
    }
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 15px;
        padding: 10px;
        border-radius: 10px;
        background: rgba(0, 30, 0, 0.7);
    }
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    .results-card {
        background: rgba(30, 58, 30, 0.7);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        animation: fadeIn 0.5s ease;
    }
    .disease-name {
        font-size: 1.5rem;
        color: #ffcc66;
        margin-bottom: 5px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin-top: 20px;
    }
    .sample-img {
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Disease labels
PLANT_DISEASE_LABELS = [
    "Apple scab", "Apple black rot", "Cedar apple rust", "Apple healthy",
    "Blueberry healthy", "Cherry healthy", "Cherry powdery mildew",
    "Corn gray leaf spot", "Corn common rust", "Corn healthy",
    "Corn northern leaf blight", "Grape black rot", "Grape esca",
    "Grape healthy", "Grape leaf blight", "Orange haunglongbing",
    "Peach bacterial spot", "Peach healthy", "Pepper bacterial spot",
    "Pepper healthy", "Potato early blight", "Potato healthy",
    "Potato late blight", "Raspberry healthy", "Soybean healthy",
    "Squash powdery mildew", "Strawberry healthy", "Strawberry leaf scorch",
    "Tomato bacterial spot", "Tomato early blight", "Tomato healthy",
    "Tomato late blight", "Tomato leaf mold", "Tomato septoria leaf spot",
    "Tomato spider mites", "Tomato target spot", 
    "Tomato yellow leaf curl virus", "Tomato mosaic virus"
]

# Sample images
SAMPLE_IMAGES = {
    "Tomato Early Blight": "https://github.com/ravirajsinh45/plant_disease_dataset/raw/master/tomato/Tomato_Early_blight.JPG",
    "Tomato Late Blight": "https://github.com/ravirajsinh45/plant_disease_dataset/raw/master/tomato/Tomato_Late_blight.JPG",
    "Tomato Healthy": "https://github.com/ravirajsinh45/plant_disease_dataset/raw/master/tomato/Healthy.JPG"
}

# API-based disease detection (simulated for demo)
def detect_disease(image):
    """Simulate disease detection with mock results"""
    try:
        # In a real implementation, this would call an API like:
        # response = requests.post(API_URL, files={'file': image})
        # results = response.json()
        
        # For demo purposes, we'll generate realistic mock results
        diseases = PLANT_DISEASE_LABELS.copy()
        np.random.shuffle(diseases)
        
        # Generate random confidence scores
        confidences = np.random.rand(len(PLANT_DISEASE_LABELS))
        confidences = confidences / confidences.sum()
        
        # Get top 3 results
        top_indices = np.argsort(confidences)[::-1][:3]
        results = []
        
        for idx in top_indices:
            results.append({
                "disease": PLANT_DISEASE_LABELS[idx],
                "confidence": confidences[idx]
            })
        
        return results
        
    except Exception as e:
        st.error(f"Error in disease detection: {str(e)}")
        return []

# Treatment recommendations
def get_treatments(disease_name):
    """Get treatment recommendations based on disease"""
    treatments = []
    
    if "Healthy" in disease_name:
        treatments = [
            "Continue regular monitoring",
            "Apply balanced NPK fertilizer",
            "Maintain proper watering schedule",
            "Ensure adequate sunlight exposure"
        ]
    elif "Early Blight" in disease_name:
        treatments = [
            "Remove infected leaves immediately",
            "Apply copper-based fungicide weekly",
            "Improve air circulation around plants",
            "Rotate crops next season"
        ]
    elif "Late Blight" in disease_name:
        treatments = [
            "Apply chlorothalonil immediately",
            "Destroy severely infected plants",
            "Avoid overhead watering",
            "Plant resistant varieties next season"
        ]
    elif "Bacterial Spot" in disease_name:
        treatments = [
            "Apply copper-based bactericides",
            "Use disease-free seeds",
            "Avoid working with plants when wet"
        ]
    elif "Septoria" in disease_name:
        treatments = [
            "Apply mancozeb fungicide",
            "Remove and destroy infected leaves",
            "Stake plants for better airflow"
        ]
    else:
        treatments = [
            "Apply neem oil spray every 7 days",
            "Remove affected plant parts",
            "Introduce beneficial insects"
        ]
    
    return treatments

# Main app
def main():
    # Initialize session state
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "results" not in st.session_state:
        st.session_state.results = None
    
    st.title("🌱 AgriScan Plant Doctor")
    st.subheader("AI-Powered Plant Disease Detection & Treatment Recommendations")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📷 Scan Plant Leaf")
        st.markdown("Upload an image of a plant leaf for disease analysis")
        
        # Image uploader
        uploaded_file = st.file_uploader("Choose a plant leaf image", 
                                       type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf", use_column_width=True)
            
            # Process button
            if st.button("🔍 Analyze Image", use_container_width=True):
                st.session_state.processing = True
                with st.spinner("Analyzing leaf..."):
                    # Simulate API processing time
                    time.sleep(2)
                    
                    # Get detection results
                    st.session_state.results = detect_disease(image)
                    st.session_state.processing = False
        
        # Sample images
        st.markdown("### Sample Images")
        cols = st.columns(3)
        for i, (name, url) in enumerate(SAMPLE_IMAGES.items()):
            with cols[i]:
                st.image(url, caption=name, use_column_width=True)
                if st.button(f"Test {name.split()[-1]}", key=f"sample_{i}"):
                    st.session_state.processing = True
                    with st.spinner("Analyzing sample..."):
                        # Load sample image
                        response = requests.get(url)
                        image = Image.open(BytesIO(response.content))
                        # Get detection results
                        st.session_state.results = detect_disease(image)
                        st.session_state.processing = False
        
        # Status indicator
        status_dot_color = "#ff9800"
        status_text = "Ready to scan"
        if st.session_state.processing:
            status_dot_color = "#FFC107"
            status_text = "Analyzing leaf..."
        
        st.markdown(
            f"""
            <div class="status-indicator">
                <div class="status-dot" style="background: {status_dot_color};"></div>
                <span>{status_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        if st.session_state.results:
            st.markdown("### 🔍 Diagnosis Results")
            primary_disease = st.session_state.results[0]["disease"]
            confidence = st.session_state.results[0]["confidence"]
            
            st.markdown(
                f"""
                <div class="results-card">
                    <div class="disease-name">{primary_disease}</div>
                    <div class="confidence">Confidence: {(confidence * 100):.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            treatments = get_treatments(primary_disease)
            st.markdown("#### 💊 Treatment Recommendations")
            st.markdown("\n".join([f"- {t}" for t in treatments]))
            
            st.markdown("#### 🌱 Prevention Tips")
            st.markdown("""
            - Rotate crops annually
            - Water early in the day
            - Inspect plants weekly
            - Sterilize tools after use
            """)
            
            # Top predictions
            st.markdown("#### 📊 Top Predictions")
            fig, ax = plt.subplots(figsize=(8, 4))
            
            diseases = [r["disease"] for r in st.session_state.results]
            confidences = [r["confidence"] for r in st.session_state.results]
            
            # Shorten long disease names
            short_names = [name[:20] + ("..." if len(name) > 20 else "") 
                          for name in diseases]
            
            bars = ax.barh(short_names, confidences, color=["#4CAF50", "#FFC107", "#FF9800"])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Confidence')
            ax.bar_label(bars, fmt='%.2f', padding=3)
            ax.set_title('Disease Confidence Scores')
            ax.invert_yaxis()  # highest confidence at top
            
            st.pyplot(fig)
            
            if st.button("🔄 New Scan", use_container_width=True):
                st.session_state.results = None
                st.session_state.processing = False
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px; opacity: 0.7;">
                <h3>Waiting for Scan Results</h3>
                <div style="font-size: 5rem; margin: 20px;">🌿</div>
                <p>Upload a plant leaf image to get diagnosis</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #aaa;'>"
        "AgriScan - AI-powered Plant Health Assistant | Demo v1.0"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

   
   
