"""
DOCWISE AI - Unified Medical Application
Combines PDF Summarization and Doctor Recommendation System
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import time
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from modules.disease_mapper import predict_specialist
from modules.doctor_filtering import get_doctors_by_specialist

# For PDF summarization
from transformers import BartForConditionalGeneration, BartTokenizer
import PyPDF2
import io

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="DOCWISE AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    * {
        font-family: 'DM Sans', sans-serif;
        box-sizing: border-box;
    }

    :root {
        --navy: #0a1628;
        --navy-mid: #112240;
        --teal: #00b4d8;
        --teal-light: #90e0ef;
        --teal-dark: #0077b6;
        --gold: #e9c46a;
        --off-white: #f0f4f8;
        --surface: #ffffff;
        --border: #dde3ec;
        --text-primary: #0a1628;
        --text-secondary: #4a5568;
        --text-muted: #8896a5;
        --shadow-sm: 0 1px 3px rgba(10,22,40,0.08), 0 1px 2px rgba(10,22,40,0.06);
        --shadow-md: 0 4px 16px rgba(10,22,40,0.10), 0 2px 6px rgba(10,22,40,0.06);
        --shadow-lg: 0 12px 40px rgba(10,22,40,0.14);
        --radius: 12px;
        --radius-lg: 18px;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--off-white) !important;
    }

    .main, [data-testid="stMain"] {
        background-color: var(--off-white) !important;
        padding: 0 !important;
    }

    .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1200px;
    }

    /* ── Header ── */
    .dw-header {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 60%, #0d3460 100%);
        border-radius: var(--radius-lg);
        padding: 2.8rem 3rem;
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }

    .dw-header::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 260px; height: 260px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(0,180,216,0.18) 0%, transparent 70%);
        pointer-events: none;
    }

    .dw-header::after {
        content: '';
        position: absolute;
        bottom: -40px; left: -40px;
        width: 200px; height: 200px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(233,196,106,0.12) 0%, transparent 70%);
        pointer-events: none;
    }

    .dw-header-inner {
        position: relative; z-index: 1;
    }

    .dw-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--teal-light);
        margin-bottom: 0.6rem;
    }

    .dw-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        line-height: 1.15;
        letter-spacing: -0.01em;
    }

    .dw-title span {
        color: var(--teal);
    }

    .dw-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: rgba(240,244,248,0.72);
        margin: 0;
        letter-spacing: 0.01em;
    }

    /* ── Section label ── */
    .dw-section-label {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--teal-dark);
        margin-bottom: 0.5rem;
    }

    .dw-section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.45rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 1.2rem 0;
    }

    /* ── Card ── */
    .dw-card {
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 2rem;
        box-shadow: var(--shadow-md);
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }

    .dw-card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }

    /* ── Upload zone ── */
    .dw-upload-zone {
        border: 2px dashed rgba(0,180,216,0.40);
        border-radius: var(--radius);
        padding: 2.5rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(0,180,216,0.04) 0%, rgba(0,119,182,0.04) 100%);
        margin: 1.2rem 0 1.5rem;
        transition: border-color 0.2s;
    }

    .dw-upload-zone:hover {
        border-color: var(--teal);
    }

    .dw-upload-icon {
        font-size: 2.2rem;
        margin-bottom: 0.6rem;
    }

    .dw-upload-zone h4 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.3rem;
    }

    .dw-upload-zone p {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* ── Summary box ── */
    .dw-summary-box {
        background: linear-gradient(135deg, #f8fbff 0%, #ffffff 100%);
        border-left: 4px solid var(--teal);
        border-radius: 0 var(--radius) var(--radius) 0;
        padding: 1.6rem 1.8rem;
        margin: 1.2rem 0;
        box-shadow: var(--shadow-sm);
    }

    .dw-summary-box .dw-summary-label {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--teal-dark);
        margin-bottom: 0.6rem;
    }

    .dw-summary-box p {
        color: var(--text-primary) !important;
        font-size: 0.95rem;
        line-height: 1.75;
        margin: 0;
    }

    .dw-empty-state {
        text-align: center;
        padding: 3rem 1.5rem;
        color: var(--text-muted);
    }

    .dw-empty-state .dw-empty-icon {
        font-size: 3rem;
        margin-bottom: 0.8rem;
        opacity: 0.5;
    }

    .dw-empty-state p {
        font-size: 0.92rem;
        margin: 0;
    }

    /* ── Metric cards ── */
    .dw-metrics-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1.2rem 0;
    }

    .dw-metric {
        background: var(--navy);
        border-radius: var(--radius);
        padding: 1.2rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .dw-metric::before {
        content: '';
        position: absolute;
        top: -20px; right: -20px;
        width: 80px; height: 80px;
        border-radius: 50%;
        background: rgba(0,180,216,0.12);
    }

    .dw-metric-value {
        font-family: 'Playfair Display', serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--teal);
        line-height: 1;
        position: relative; z-index: 1;
    }

    .dw-metric-label {
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: rgba(240,244,248,0.65);
        margin-top: 0.35rem;
        position: relative; z-index: 1;
    }

    /* ── Doctor card ── */
    .dw-doctor-card {
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: 1.5rem 1.8rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1.5rem;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }

    .dw-doctor-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, var(--teal) 0%, var(--teal-dark) 100%);
        border-radius: 4px 0 0 4px;
        opacity: 0;
        transition: opacity 0.25s;
    }

    .dw-doctor-card:hover {
        box-shadow: var(--shadow-md);
        border-color: rgba(0,180,216,0.30);
        transform: translateX(4px);
    }

    .dw-doctor-card:hover::before {
        opacity: 1;
    }

    .dw-doctor-rank {
        font-family: 'Playfair Display', serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--border);
        min-width: 2.5rem;
        line-height: 1;
    }

    .dw-doctor-info h3 {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.3rem;
    }

    .dw-doctor-meta {
        font-size: 0.83rem;
        color: var(--text-secondary);
        margin: 0.2rem 0;
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
    }

    .dw-doctor-meta span {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    .dw-rating-badge {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-mid) 100%);
        color: var(--teal);
        border-radius: 50%;
        width: 58px; height: 58px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(10,22,40,0.20);
    }

    .dw-rating-badge .dw-rating-num {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1;
    }

    .dw-rating-badge .dw-rating-label {
        font-size: 0.55rem;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: rgba(240,244,248,0.55);
        margin-top: 0.15rem;
    }

    /* ── Specialist badge ── */
    .dw-specialist-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, rgba(0,180,216,0.12) 0%, rgba(0,119,182,0.08) 100%);
        border: 1px solid rgba(0,180,216,0.28);
        border-radius: 100px;
        padding: 0.45rem 1.1rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--teal-dark);
        margin-bottom: 1.5rem;
    }

    /* ── How it works panel ── */
    .dw-how-panel {
        background: linear-gradient(135deg, var(--navy) 0%, #0d3460 100%);
        border-radius: var(--radius-lg);
        padding: 2rem;
        margin-top: 0.5rem;
    }

    .dw-how-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin: 0 0 1.2rem;
    }

    .dw-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 1rem;
        color: rgba(240,244,248,0.82);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    .dw-step:last-child {
        margin-bottom: 0;
    }

    .dw-step-num {
        background: rgba(0,180,216,0.20);
        color: var(--teal);
        border-radius: 50%;
        width: 26px; height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
        margin-top: 1px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--teal-dark) 0%, var(--navy-mid) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius) !important;
        padding: 0.78rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s ease !important;
        width: 100%;
        box-shadow: 0 4px 14px rgba(0,119,182,0.28) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, var(--teal) 0%, var(--teal-dark) 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,180,216,0.35) !important;
    }

    /* ── Inputs ── */
    .stTextInput input {
        border-radius: var(--radius) !important;
        border: 1.5px solid var(--border) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.92rem !important;
        padding: 0.65rem 1rem !important;
        background: #ffffff !important;
        color: var(--text-primary) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stTextInput input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 3px rgba(0,180,216,0.12) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy) 0%, #0d3460 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06) !important;
    }

    [data-testid="stSidebar"] * {
        color: rgba(240,244,248,0.85) !important;
    }

    .dw-sidebar-brand {
        text-align: center;
        padding: 1.8rem 1rem 1.2rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 1rem;
    }

    .dw-sidebar-logo {
        font-family: 'Playfair Display', serif;
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff !important;
        letter-spacing: 0.04em;
    }

    .dw-sidebar-logo span {
        color: var(--teal) !important;
    }

    .dw-sidebar-tagline {
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: rgba(144,224,239,0.65) !important;
        margin-top: 0.25rem;
    }

    .dw-sidebar-footer {
        font-size: 0.72rem;
        text-align: center;
        color: rgba(240,244,248,0.35) !important;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 1rem;
        letter-spacing: 0.06em;
    }

    /* ── Alerts ── */
    [data-testid="stSuccess"] {
        background: rgba(0,180,216,0.08) !important;
        border: 1px solid rgba(0,180,216,0.25) !important;
        border-radius: var(--radius) !important;
        color: var(--teal-dark) !important;
    }

    [data-testid="stWarning"] {
        border-radius: var(--radius) !important;
    }

    [data-testid="stError"] {
        border-radius: var(--radius) !important;
    }

    [data-testid="stInfo"] {
        border-radius: var(--radius) !important;
    }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
        margin: 2rem 0 !important;
    }

    /* ── Sliders ── */
    .stSlider [data-testid="stSlider"] {
        accent-color: var(--teal) !important;
    }

    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============ LOAD MODELS (CACHED) ============
@st.cache_resource
def load_bart_model():
    """Load BART model for PDF summarization"""
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_doctor_data():
    """Load doctor profiles CSV"""
    try:
        df = pd.read_csv("data/doctor_profiles.csv")
        return df
    except:
        return None

# ============ PDF FUNCTIONS ============
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

def generate_summary(text, tokenizer, model, max_length=200, min_length=50):
    """Generate summary using BART model - optimized for speed"""
    try:
        # Truncate text for faster processing
        inputs = tokenizer.encode(
            "summarize: " + text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate summary with optimized parameters for speed
        summary_ids = model.generate(
    inputs,
    max_length=max_length,
    min_length=min_length,
    num_beams=2,
    length_penalty=1.5,
    early_stopping=True,
    no_repeat_ngram_size=3,
    forced_bos_token_id=tokenizer.bos_token_id
)
        
        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# ============ DOCTOR DASHBOARD ============
def doctor_dashboard():
    """Doctor Dashboard - PDF Summarization"""
    st.markdown("""
    <div class="dw-header">
        <div class="dw-header-inner">
            <div class="dw-eyebrow">Doctor Workspace</div>
            <h1 class="dw-title">Medical Report <span>Summariser</span></h1>
            <p class="dw-subtitle">Upload a clinical PDF and get a precise AI-generated summary in seconds.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load BART model
    with st.spinner("Loading AI model…"):
        tokenizer, model = load_bart_model()
    
    st.success("✅ AI model loaded and ready")
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="dw-section-label" style="font-size:0.70rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#0077b6;margin-bottom:0.5rem;">Step 1</div>', unsafe_allow_html=True)
        st.markdown('<div class="dw-section-title" style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:600;color:#0a1628;margin:0 0 1.2rem 0;">Upload Report</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="dw-upload-zone">
            <div class="dw-upload-icon">📄</div>
            <h4 style="color:#0a1628;">Drag & drop or browse</h4>
            <p style="color:#4a5568;">Supports text-based PDF medical reports</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a text-based medical PDF report",
            label_visibility="collapsed"
        )
        
        # Summarization parameters
        with st.expander("⚙️ Summarisation Settings"):
            max_length = st.slider("Maximum Summary Length", 50, 5000, 200, 10)
            min_length = st.slider("Minimum Summary Length", 10, 500, 50, 5)
    
    with col2:
        st.markdown('<div class="dw-section-label" style="font-size:0.70rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#0077b6;margin-bottom:0.5rem;">Step 2</div>', unsafe_allow_html=True)
        st.markdown('<div class="dw-section-title" style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:600;color:#0a1628;margin:0 0 1.2rem 0;">Generated Summary</div>', unsafe_allow_html=True)
        
        if uploaded_pdf is not None:
            st.info(f"📎 **{uploaded_pdf.name}** — {uploaded_pdf.size / 1024:.2f} KB")
            
            if st.button("🚀 Generate Summary", use_container_width=True):
                start_time = time.time()
                
                with st.spinner("🔄 Extracting text from PDF…"):
                    try:
                        final_text = extract_text_from_pdf(uploaded_pdf)
                        word_count = len(final_text.split())
                        st.caption(f"Words detected: {word_count:,}")
                    except Exception as e:
                        st.error(f"❌ Error reading PDF: {str(e)}")
                        return
                
                with st.spinner("🤖 Generating AI summary…"):
                    try:
                        summary = generate_summary(
                            final_text,
                            tokenizer,
                            model,
                            max_length,
                            min_length
                        )
                        
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        st.markdown(f"""
                        <div class="dw-summary-box">
                            <div class="dw-summary-label" style="font-size:0.68rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#0077b6;margin-bottom:0.6rem;">AI Summary</div>
                            <p style="color:#0a1628 !important;font-size:0.95rem;line-height:1.75;margin:0;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        summary_words = len(summary.split())
                        compression = round((1 - summary_words / word_count) * 100, 1)
                        
                        st.markdown(f"""
                        <div class="dw-metrics-row">
                            <div class="dw-metric">
                                <div class="dw-metric-value">{word_count:,}</div>
                                <div class="dw-metric-label">Original Words</div>
                            </div>
                            <div class="dw-metric">
                                <div class="dw-metric-value">{summary_words}</div>
                                <div class="dw-metric-label">Summary Words</div>
                            </div>
                            <div class="dw-metric">
                                <div class="dw-metric-value">{compression}%</div>
                                <div class="dw-metric-label">Compression</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.success(f"⏱️ Completed in {processing_time:.2f}s")
                        
                        st.download_button(
                            "📥 Download Summary",
                            summary,
                            file_name="medical_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
        else:
            st.markdown("""
            <div class="dw-empty-state">
                <div class="dw-empty-icon" style="font-size:3rem;margin-bottom:0.8rem;opacity:0.5;">📋</div>
                <p style="color:#4a5568;font-size:0.92rem;margin:0;">No summary generated yet.<br>Upload a PDF report to get started.</p>
            </div>
            """, unsafe_allow_html=True)

# ============ PATIENT DASHBOARD ============
def patient_dashboard():
    """Patient Dashboard - Doctor Recommendation"""
    st.markdown("""
    <div class="dw-header">
        <div class="dw-header-inner">
            <div class="dw-eyebrow">Patient Portal</div>
            <h1 class="dw-title">Find Your <span>Specialist</span></h1>
            <p class="dw-subtitle">Describe your symptoms and we'll match you with the right doctor, instantly.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="dw-section-label" style="font-size:0.70rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#0077b6;margin-bottom:0.5rem;">Search</div>', unsafe_allow_html=True)
        st.markdown('<div class="dw-section-title" style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:600;color:#0a1628;margin:0 0 1.2rem 0;">Your Symptoms</div>', unsafe_allow_html=True)
        
        disease = st.text_input(
            "Symptoms or diagnosis",
            placeholder="e.g., diabetes, headache, fever",
            help="Enter the condition or symptoms you're experiencing"
        )
        
        location = st.text_input(
            "Preferred location",
            placeholder="e.g., Chennai, Mumbai, Delhi",
            help="Enter your preferred location for doctor search"
        )
        
        search_clicked = st.button("🔎 Find Doctors", use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="dw-how-panel">
            <div class="dw-how-title" style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:600;color:#ffffff;margin:0 0 1.2rem;">How It Works</div>
            <div class="dw-step" style="display:flex;align-items:flex-start;gap:1rem;margin-bottom:1rem;">
                <div class="dw-step-num" style="background:rgba(0,180,216,0.20);color:#00b4d8;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;">1</div>
                <div style="color:rgba(240,244,248,0.85);font-size:0.88rem;line-height:1.55;">Enter your symptoms or diagnosed condition in the search field.</div>
            </div>
            <div class="dw-step" style="display:flex;align-items:flex-start;gap:1rem;margin-bottom:1rem;">
                <div class="dw-step-num" style="background:rgba(0,180,216,0.20);color:#00b4d8;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;">2</div>
                <div style="color:rgba(240,244,248,0.85);font-size:0.88rem;line-height:1.55;">Specify your preferred city or area for localised results.</div>
            </div>
            <div class="dw-step" style="display:flex;align-items:flex-start;gap:1rem;margin-bottom:1rem;">
                <div class="dw-step-num" style="background:rgba(0,180,216,0.20);color:#00b4d8;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;">3</div>
                <div style="color:rgba(240,244,248,0.85);font-size:0.88rem;line-height:1.55;">Our AI maps your condition to the right specialist category.</div>
            </div>
            <div class="dw-step" style="display:flex;align-items:flex-start;gap:1rem;">
                <div class="dw-step-num" style="background:rgba(0,180,216,0.20);color:#00b4d8;border-radius:50%;width:26px;height:26px;display:flex;align-items:center;justify-content:center;font-size:0.75rem;font-weight:700;flex-shrink:0;">4</div>
                <div style="color:rgba(240,244,248,0.85);font-size:0.88rem;line-height:1.55;">Browse top-rated doctors ranked by experience and patient reviews.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if search_clicked and disease:
        st.markdown("---")
        
        with st.spinner("Matching you with the best specialists…"):
            specialist = predict_specialist(disease)
            
            if specialist:
                st.markdown(f"""
                <div class="dw-specialist-badge">
                    🩺 <span style="color:#0077b6;">Recommended Specialist: <strong style="color:#0a1628;">{specialist}</strong></span>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    doctors_df = get_doctors_by_specialist(
                        specialist, 
                        location=location if location else None,
                        min_experience=2,
                        min_rating=3.5
                    )
                    
                    if not doctors_df.empty:
                        doctors_df = doctors_df.sort_values(by="Rating", ascending=False)
                        
                        st.markdown('<div class="dw-section-label" style="font-size:0.70rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#0077b6;margin-bottom:0.5rem;">Results</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="dw-section-title" style="font-family:Playfair Display,serif;font-size:1.45rem;font-weight:600;color:#0a1628;margin:0 0 1.2rem 0;">Top {len(doctors_df)} Doctors Found</div>', unsafe_allow_html=True)
                        
                        for idx, (_, doctor) in enumerate(doctors_df.iterrows(), 1):
                            st.markdown(f"""
                            <div class="dw-doctor-card">
                                <div class="dw-doctor-rank" style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:#dde3ec;min-width:2.5rem;line-height:1;">#{idx}</div>
                                <div class="dw-doctor-info" style="flex:1;">
                                    <h3 style="font-size:1.05rem;font-weight:600;color:#0a1628;margin:0 0 0.3rem;">{doctor['Name']}</h3>
                                    <div class="dw-doctor-meta" style="font-size:0.83rem;color:#4a5568;margin:0.2rem 0;display:flex;flex-wrap:wrap;gap:0.8rem;">
                                        <span style="color:#4a5568;">👨‍⚕️ {doctor['Specialist']}</span>
                                        <span style="color:#4a5568;">💼 {doctor['Experience']} yrs exp</span>
                                        <span style="color:#4a5568;">🏢 {doctor['Location']}</span>
                                        <span style="color:#4a5568;">📞 {doctor['Contact']}</span>
                                    </div>
                                </div>
                                <div class="dw-rating-badge" style="background:linear-gradient(135deg,#0a1628 0%,#112240 100%);border-radius:50%;width:58px;height:58px;display:flex;flex-direction:column;align-items:center;justify-content:center;flex-shrink:0;box-shadow:0 4px 12px rgba(10,22,40,0.20);">
                                    <div style="font-family:'Playfair Display',serif;font-size:1.15rem;font-weight:700;color:#00b4d8;line-height:1;">{doctor['Rating']}</div>
                                    <div style="font-size:0.55rem;font-weight:600;letter-spacing:0.10em;text-transform:uppercase;color:rgba(240,244,248,0.65);margin-top:0.15rem;">Rating</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    else:
                        st.warning("⚠️ No suitable doctors found in your area. Try broadening your location.")
                        
                except Exception as e:
                    st.error(f"❌ Error fetching doctors: {str(e)}")
            else:
                st.error("❌ Condition not found in our database. Please try a different search term.")
    
    elif search_clicked and not disease:
        st.warning("⚠️ Please enter a disease or symptom to search for doctors.")

# ============ MAIN APP ============
def main():
    """Main application - no authentication required"""
    
    with st.sidebar:
        st.markdown("""
        <div class="dw-sidebar-brand">
            <div class="dw-sidebar-logo">DOC<span>WISE</span> AI</div>
            <div class="dw-sidebar-tagline">Medical Intelligence Platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["👨‍⚕️ Doctor", "🧑‍🤝‍🧑 Patient"],
            icons=["hospital", "people"],
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "rgba(144,224,239,0.85)", "font-size": "16px"},
                "nav-link": {
                    "font-size": "0.92rem",
                    "font-weight": "500",
                    "text-align": "left",
                    "margin": "3px 0",
                    "padding": "0.65rem 1rem",
                    "border-radius": "10px",
                    "--hover-color": "rgba(255,255,255,0.10)",
                    "color": "rgba(240,244,248,0.80)",
                    "letter-spacing": "0.01em",
                },
                "nav-link-selected": {
                    "background-color": "rgba(0,180,216,0.18)",
                    "color": "#ffffff",
                    "font-weight": "600",
                    "border": "1px solid rgba(0,180,216,0.25)",
                },
            }
        )
        
        st.markdown("""
        <div class="dw-sidebar-footer">DOCWISE AI © 2025</div>
        """, unsafe_allow_html=True)
    
    if selected == "👨‍⚕️ Doctor":
        doctor_dashboard()
    else:
        patient_dashboard()

if __name__ == "__main__":
    main()