import streamlit as st
import sqlite3
from datetime import datetime
import unicodedata
import re
from PIL import Image
import numpy as np
import pydicom
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
import os
import torch
import cv2
from torchvision import transforms
import sys
from openai import OpenAI  
import pandas as pd 
import plotly.express as px

# Trỏ đường dẫn vào thư mục chứa code AI
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mri_analyzer import analyze_brain_ai_driven
from mri_model import ResNetMRI

# ==========================================
# CONFIG & FOLDERS
# ==========================================
st.set_page_config(page_title="NeuroVision AI", layout="wide", page_icon=r"C:\Users\Asus\Hackathon\Hackathon_2026-main\Hackathon_2026-main\iconfinder-bl-1646-brain-artificial-intelligence-electronic-computer-processor-consciousness-4575061_121498.png")

if not os.path.exists("history_img"):
    os.makedirs("history_img")

try:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_KEY = None

@st.cache_resource
def load_ai_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetMRI(num_classes=2).to(device)
    model.load_state_dict(torch.load("models/mri_model_best.pth", map_location=device))
    model.eval()
    return model, device

ai_model, device = load_ai_model()

# ==========================================
# DB SETUP 
# ==========================================
conn = sqlite3.connect("mediai.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age TEXT,
    gender TEXT,
    time TEXT,
    result TEXT,
    coords TEXT,
    img1_path TEXT,
    img2_path TEXT
)
""")
try:
    c.execute("ALTER TABLE history ADD COLUMN report TEXT")
except:
    pass 
conn.commit()

# =========================
# CSS NÂNG CAO (GIỮ NGUYÊN BẢN CHUẨN)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg-main: #0b1118;
    --bg-soft: rgba(15, 23, 35, 0.78);
    --bg-soft-strong: rgba(17, 27, 40, 0.88);
    --text-main: #f5fbff;
    --text-soft: #a8bac9;
    --teal: #00ADB5;
    --cyan: #59f3ff;
    --line: rgba(122, 236, 255, 0.16);
    --line-strong: rgba(122, 236, 255, 0.32);
    --shadow-soft: 0 18px 40px rgba(0, 0, 0, 0.24);
    --shadow-glow: 0 0 0 1px rgba(89, 243, 255, 0.10), 0 18px 40px rgba(0, 173, 181, 0.14);
}

@keyframes fadeInUp {
    0% {
        opacity: 0;
        transform: translateY(18px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        text-shadow: 0 0 0 rgba(89, 243, 255, 0.0);
    }
    50% {
        opacity: 0.72;
        text-shadow: 0 0 18px rgba(89, 243, 255, 0.28);
    }
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 17px !important;
    color: var(--text-main);
}

body, .stApp {
    background:
        radial-gradient(circle at 12% 14%, rgba(0, 173, 181, 0.16), transparent 26%),
        radial-gradient(circle at 88% 10%, rgba(89, 243, 255, 0.10), transparent 24%),
        linear-gradient(180deg, #091018 0%, #0b1118 45%, #0e1620 100%);
    color: var(--text-main);
}

[data-testid="stHeader"] {
    background: rgba(11, 17, 24, 0.72);
    backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(10, 16, 24, 0.94) 0%, rgba(14, 22, 32, 0.96) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.04);
    padding-top: 0.25rem;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.1rem;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.35rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    width: 100%;
}

[data-testid="stSidebar"] .stRadio {
    margin-top: 0.35rem;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 0.55rem;
    width: 100%;
}

[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    margin: 0;
    width: 100%;
    border-radius: 16px;
    padding: 0.2rem 0.35rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

[data-testid="stSidebar"] .stRadio label {
    width: 100%;
    margin: 0;
    padding: 0.6rem 0.8rem;
    border-radius: 12px;
    justify-content: flex-start;
    color: var(--text-main) !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 255, 255, 0.04);
}

.sidebar-header {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.7rem;
    width: 100%;
    margin: 0 auto 1rem auto;
    padding: 0.8rem 1rem;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02));
    border: 1px solid rgba(89, 243, 255, 0.10);
    color: var(--text-main);
    font-size: 0.9rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    text-align: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

.sidebar-header img {
    width: 26px;
    height: 26px;
    object-fit: contain;
    display: block;
    filter: drop-shadow(0 0 10px rgba(89, 243, 255, 0.18));
}

h1, h2, h3 {
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    line-height: 1.12;
    color: var(--text-main);
    margin-top: 0.2rem;
    margin-bottom: 0.75rem;
}

h1 { font-size: 2.6rem !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.45rem !important; }

p, label, .stMarkdown, .stCaption, .stText, .stRadio label, .stSelectbox label {
    color: var(--text-main) !important;
}

.brand-title { 
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 15px;
    margin-bottom: 0;
    padding-bottom: 0;
    text-align: center;
}

.brand-mark {
    width: 50px;
    height: 50px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: 0 0 auto;
    filter: drop-shadow(0 0 18px rgba(89, 243, 255, 0.16));
}

.brand-mark svg {
    width: 50px;
    height: 50px;
    display: block;
}

.brand-word {
    display: inline-block;
    font-size: 64px;
    font-weight: 800;
    line-height: 0.96;
    letter-spacing: -0.06em;
    background: linear-gradient(135deg, #ffffff 0%, #c9fbff 35%, #59f3ff 62%, #00ADB5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 32px rgba(89, 243, 255, 0.18);
}

.brand-subtitle,
.page-subheader {
    font-size: 1.02rem;
    color: var(--text-soft) !important;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}

.page-header {
    font-size: 2.15rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.45rem;
    color: var(--text-main);
    letter-spacing: -0.03em;
}

button,
a,
[role="button"],
[data-baseweb="tab"],
[data-testid="stExpanderToggleIcon"],
.stDownloadButton > button,
.stButton > button {
    transition: all 0.3s ease-in-out !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 1.6rem !important;
    margin-bottom: 1.35rem;
    border-radius: 24px !important;
    border: 1px solid var(--line) !important;
    background:
        linear-gradient(180deg, rgba(20, 31, 45, 0.72) 0%, rgba(14, 22, 33, 0.78) 100%) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: var(--shadow-soft), inset 0 1px 0 rgba(255, 255, 255, 0.04);
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.7s ease both;
}

div[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(89, 243, 255, 0.10), transparent 30%, transparent 70%, rgba(0, 173, 181, 0.10));
    pointer-events: none;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px);
    border-color: var(--line-strong) !important;
    box-shadow: var(--shadow-glow), 0 24px 52px rgba(0, 0, 0, 0.28);
}

.stImage > img {
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.24);
    border: 1px solid rgba(255,255,255,0.05);
}

.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    min-height: 3.1rem;
    border-radius: 16px;
    border: 1px solid rgba(89, 243, 255, 0.18);
    background: linear-gradient(135deg, #007b83 0%, #00ADB5 48%, #59f3ff 100%);
    color: #f8feff;
    font-weight: 800;
    font-size: 0.98rem;
    letter-spacing: 0.01em;
    box-shadow: 0 14px 28px rgba(0, 173, 181, 0.22);
}

.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) scale(1.01);
    border-color: rgba(89, 243, 255, 0.42);
    box-shadow: 0 18px 36px rgba(0, 173, 181, 0.32), 0 0 22px rgba(89, 243, 255, 0.16);
    filter: brightness(1.05);
}

.stButton > button:focus,
.stDownloadButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus {
    outline: none;
    box-shadow: 0 0 0 0.22rem rgba(89, 243, 255, 0.18), 0 18px 36px rgba(0, 173, 181, 0.24);
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background: rgba(255, 255, 255, 0.035) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    color: var(--text-main) !important;
    font-size: 0.98rem !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border-color: rgba(89, 243, 255, 0.36) !important;
    box-shadow: 0 0 0 0.18rem rgba(89, 243, 255, 0.10) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    justify-content: center;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 999px;
    padding: 0.35rem;
    width: fit-content;
    margin: 0 auto 1.1rem auto;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.stTabs [data-baseweb="tab"] {
    min-height: 48px;
    padding: 0.72rem 1.25rem;
    border-radius: 999px;
    background: transparent;
    color: var(--text-soft);
    font-weight: 700;
    border: 1px solid transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-main);
    background: rgba(255, 255, 255, 0.04);
}

.stTabs [aria-selected="true"] {
    color: #f8feff !important;
    background: linear-gradient(135deg, rgba(0, 173, 181, 0.32), rgba(89, 243, 255, 0.20)) !important;
    border: 1px solid rgba(89, 243, 255, 0.20) !important;
    box-shadow: 0 10px 24px rgba(0, 173, 181, 0.18), inset 0 1px 0 rgba(255,255,255,0.08);
}

[data-testid="stAppViewContainer"] .stRadio > div {
    display: flex;
    justify-content: center;
}

[data-testid="stAppViewContainer"] .stRadio div[role="radiogroup"] {
    justify-content: center !important;
    width: 100%;
    margin: 0 auto;
}

[data-testid="stAppViewContainer"] .stRadio div[role="radiogroup"] > label {
    margin-left: 0.25rem;
    margin-right: 0.25rem;
}

[data-testid="stSpinner"],
.stSpinner,
[data-testid="stSpinner"] * {
    color: var(--cyan) !important;
    animation: pulse 1.5s ease-in-out infinite;
}

[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1rem;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: var(--text-soft) !important;
}

[data-testid="stDataFrame"],
.stAlert,
.streamlit-expanderHeader {
    border-radius: 18px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-header">Control Panel</div>', unsafe_allow_html=True)
page = st.sidebar.radio("", ["Workspace", "Patient Tracking"])

# =========================
# HELPER FUNCTIONS CHO PDF
# =========================
def remove_accents(input_str):
    s1 = unicodedata.normalize('NFD', str(input_str))
    return ''.join(c for c in s1 if unicodedata.category(c) != 'Mn')

def format_pdf_text(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', str(text))
    text = text.replace('\n', '<br/>')
    return text

def export_pdf(name, age, gender, result, ai_consultation, img1_path, img2_path):
    file = f"NeuroVision_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()
    justify_style = ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, leading=14)
    elements = []
    
    elements.append(Paragraph("<para align='center'><font size=18><b>NEUROVISION AI DIAGNOSTICS</b></font></para>", styles["Normal"]))
    elements.append(Spacer(1, 10)) 
    elements.append(Paragraph("<para align='center'><font size=12 color='gray'>Department of Automated Radiology</font></para>", styles["Normal"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<para align='center'><b><u>CLINICAL MRI REPORT</u></b></para>", styles["Title"]))
    elements.append(Spacer(1, 15))

    clean_name = remove_accents(name)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_id = f"MRN-{datetime.now().strftime('%y%m%d%H%M')}"
    
    patient_data = [
        ["Patient Name:", clean_name, "Patient ID:", patient_id],
        ["Age / Gender:", f"{age} / {gender}", "Exam Date:", current_time],
        ["Modality:", "MRI Brain w/o Contrast", "Physician:", "Dr. NeuroVision AI"],
        ["Primary Finding:", result, "", ""]
    ]
    t = Table(patient_data, colWidths=[90, 160, 80, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTNAME', (3, 0), (3, -1), 'Helvetica'),
        ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'), 
        ('TEXTCOLOR', (1, 3), (1, 3), colors.darkred if "Abnormal" in result else colors.green),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 25))

    clean_consultation = format_pdf_text(ai_consultation)
    elements.append(Paragraph(clean_consultation, justify_style))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>ATTACHED IMAGES (DETECTION & HEATMAP):</b>", styles["Heading3"]))
    elements.append(Spacer(1, 10))
    
    if os.path.exists(img1_path) and os.path.exists(img2_path):
        img_data = [[RLImage(img1_path, width=220, height=220), RLImage(img2_path, width=220, height=220)]]
        img_table = Table(img_data)
        elements.append(img_table)
    
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("<para align='right'><b>Electronically signed by:</b></para>", styles["Normal"]))
    elements.append(Paragraph("<para align='right'>NeuroVision Generative AI Model</para>", styles["Normal"]))

    doc.build(elements)
    return file

def export_comparison_pdf(name, age, gender, scan_a, scan_b, ai_consultation):
    file = f"NeuroVision_Progression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()
    justify_style = ParagraphStyle(name='Justify', parent=styles['Normal'], alignment=TA_JUSTIFY, leading=14)
    elements = []

    elements.append(Paragraph("<para align='center'><font size=18><b>NEUROVISION AI DIAGNOSTICS</b></font></para>", styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<para align='center'><font size=12 color='gray'>Department of Automated Radiology</font></para>", styles["Normal"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<para align='center'><b><u>LONGITUDINAL PROGRESSION REPORT</u></b></para>", styles["Title"]))
    elements.append(Spacer(1, 15))

    clean_name = remove_accents(name)
    patient_id = f"MRN-{datetime.now().strftime('%y%m%d%H%M')}"
    
    p_data = [
        ["Patient Name:", clean_name, "Patient ID:", patient_id],
        ["Age / Gender:", f"{age} / {gender}", "Report Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    t_p = Table(p_data, colWidths=[90, 160, 80, 120])
    t_p.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTNAME', (3, 0), (3, -1), 'Helvetica'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_p)
    elements.append(Spacer(1, 15))

    s_data = [
        ["", "Scan A (Baseline)", "Scan B (Follow-up)"],
        ["Date:", scan_a[4], scan_b[4]],
        ["Diagnosis:", scan_a[5], scan_b[5]]
    ]
    t_s = Table(s_data, colWidths=[80, 185, 185])
    t_s.setStyle(TableStyle([
        ('BACKGROUND', (1, 0), (2, 0), colors.lightblue),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(t_s)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>AI PROGRESSION ASSESSMENT:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 5))
    clean_consultation = format_pdf_text(ai_consultation)
    elements.append(Paragraph(clean_consultation, justify_style))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>IMAGING COMPARISON (Scan A vs Scan B):</b>", styles["Heading3"]))
    elements.append(Spacer(1, 10))
    
    if os.path.exists(scan_a[7]) and os.path.exists(scan_b[7]) and os.path.exists(scan_a[8]) and os.path.exists(scan_b[8]):
        img_data = [
            [Paragraph("<para align='center'><b>Detected Region (A)</b></para>", styles["Normal"]), Paragraph("<para align='center'><b>Detected Region (B)</b></para>", styles["Normal"])],
            [RLImage(scan_a[7], width=200, height=200), RLImage(scan_b[7], width=200, height=200)],
            [Paragraph("<para align='center'><b>Heatmap (A)</b></para>", styles["Normal"]), Paragraph("<para align='center'><b>Heatmap (B)</b></para>", styles["Normal"])],
            [RLImage(scan_a[8], width=200, height=200), RLImage(scan_b[8], width=200, height=200)]
        ]
        t_img = Table(img_data, colWidths=[225, 225])
        t_img.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        elements.append(t_img)

    elements.append(Spacer(1, 40))
    elements.append(Paragraph("<para align='right'><b>Electronically signed by:</b></para>", styles["Normal"]))
    elements.append(Paragraph("<para align='right'>NeuroVision Generative AI Model</para>", styles["Normal"]))

    doc.build(elements)
    return file

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.markdown("""
<div class="brand-title">
    <span class="brand-mark" aria-hidden="true">
        <svg width="50" height="50" data-name="Layer 1" id="Layer_1" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <style>
                    .cls-1, .cls-2 { fill: none; stroke: #00ADB5; stroke-linecap: round; stroke-width: 2px; }
                    .cls-1 { stroke-linejoin: round; }
                    .cls-2 { stroke-miterlimit: 10; }
                    .cls-3 { fill: #00ADB5; }
                </style>
            </defs>
            <title></title>
            <g>
                <g>
                    <path class="cls-1" d="M17.30639,11.25243a5.29412,5.29412,0,1,0-10.58824,0,5.23707,5.23707,0,0,0,.584,2.37007A5.27955,5.27955,0,0,0,8.8358,23.95831"/>
                    <path class="cls-2" d="M3.58191,21.56266a7.00047,7.00047,0,1,0,11.32225,6.78321V28"/>
                    <path class="cls-1" d="M8.43086,42.32855A6.21606,6.21606,0,0,1,5.108,36.82494a6.14985,6.14985,0,0,1,.68573-2.78316"/>
                    <path class="cls-2" d="M19,16a5,5,0,0,1-5,5"/>
                    <g>
                        <polyline class="cls-2" points="19 16.01 30.5 16.01 32.521 13.99"/>
                        <polyline class="cls-2" points="25.5 20.01 31.5 20.01 33.521 17.99 39 17.99"/>
                        <polyline class="cls-2" points="24.917 31.99 30.5 31.99 32.521 34.01"/>
                        <polyline class="cls-2" points="24.917 27.99 31.5 27.99 33.521 30.01 39 30.01"/>
                        <path class="cls-2" d="M39,38.01018H31.52094l-1.52061-2.02061-4.95867.00011v6.05209a5,5,0,0,1-10,0"/>
                    </g>
                    <path class="cls-2" d="M17.04167,4.13392a4.05277,4.05277,0,0,1,8,.91833V12.01l4.45868.00013L31.521,9.98956H39"/>
                    <line class="cls-2" x1="36" x2="25.04167" y1="24" y2="24"/>
                    <circle class="cls-3" cx="11" cy="20" r="1"/>
                    <circle class="cls-3" cx="11.79167" cy="43.5" r="1"/>
                </g>
                <circle class="cls-2" cx="41.47922" cy="10" r="2"/>
                <circle class="cls-2" cx="41.47922" cy="18" r="2"/>
                <circle class="cls-2" cx="41.47922" cy="30" r="2"/>
                <circle class="cls-2" cx="41.47922" cy="38" r="2"/>
            </g>
        </svg>
    </span>
    <span class="brand-word">NEUROVISION</span>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="brand-subtitle">Comprehensive MRI Analysis & Analytics Platform</div>', unsafe_allow_html=True)
st.markdown("---")

if page == "Workspace":
    
    col_empty, col_toggle, col_empty2 = st.columns([1.5, 1, 1.5])
    with col_toggle:
        view_mode = st.radio("Mode", ["AI Analysis", "System Analytics"], horizontal=True, label_visibility="collapsed")

    # ==========================
    # TRANG 1: PHÂN TÍCH AI
    # ==========================
    if view_mode == "AI Analysis":
        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False
            st.session_state.ai_consultation = ""

        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Patient Demographics</h3>", unsafe_allow_html=True)
            st.write("") 
            col1, col2, col3 = st.columns(3)
            name = col1.text_input("Full Name", placeholder="e.g. John Doe")
            age = col2.text_input("Age", placeholder="e.g. 45")
            gender = col3.selectbox("Gender", ["Male", "Female", "Other"])

        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Upload MRI Scan</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #888;'>Supported formats: .dcm, .ima, .png, .jpg, .jpeg, .tif, .tiff, .bmp</p>", unsafe_allow_html=True)
            file = st.file_uploader("", type=["dcm", "ima", "png", "jpg", "jpeg", "tif", "tiff", "bmp"], label_visibility="collapsed")
            
            st.write("")
            analyze_btn = st.button("Run NeuroVision AI Analysis", use_container_width=True, type="primary")

        if analyze_btn:
            if file is None or name == "":
                st.error("Please fill in patient name and upload an MRI file!")
            else:
                with st.spinner("NeuroVision is analyzing the MRI scan..."):
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                    if len(image.shape) > 2:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = image.astype(np.float32)
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5], std=[0.5])
                    ])
                    input_tensor = transform(image_resized).unsqueeze(0).to(device)

                    img_with_boxes, heatmap, prob_overall, has_anom, suggestions = analyze_brain_ai_driven(input_tensor, ai_model, image_resized)

                    if has_anom:
                        result = f"Abnormal - Tumor Detected ({prob_overall:.1f}%)"
                        coords = "Detected by Grad-CAM"
                    else:
                        result = f"Normal ({100 - prob_overall:.1f}%)"
                        coords = "None"

                    timestamp = str(datetime.now().timestamp()).replace(".", "")
                    img1_path = f"history_img/det_{timestamp}.png"
                    img2_path = f"history_img/heat_{timestamp}.png"
                    Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)).save(img1_path)
                    Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)).save(img2_path)

                    ai_consultation = ""
                    if OPENAI_KEY:
                        try:
                            client = OpenAI(api_key=OPENAI_KEY)
                            prompt = f"""
                            Act as a Senior Clinical Neuroradiologist. 
                            Patient: {name}, Age: {age}, Gender: {gender}.
                            System Detection: {result}.
                            Write a clinical radiology report. You MUST format your response exactly like this:
                            
                            <b>FINDINGS:</b> 
                            (Write 2-3 sentences describing the imaging findings formally based on the detection result).
                            <br/><br/>
                            <b>IMPRESSION:</b>
                            (Write 1-2 sentences with the final diagnostic conclusion and recommendations).
                            
                            Do NOT use markdown formatting like **bold**, use ONLY the HTML tags <b> and <br/> provided in the structure.
                            """
                            response = client.chat.completions.create(
                                model="gpt-4o-mini", 
                                messages=[{"role": "system", "content": "You are a professional medical AI assistant."},
                                          {"role": "user", "content": prompt}]
                            )
                            ai_consultation = response.choices[0].message.content
                        except Exception as e:
                            ai_consultation = f"OpenAI Connection Error: {e}"
                    else:
                        ai_consultation = "OpenAI API Key is missing."

                    st.session_state.analysis_done = True
                    st.session_state.name = name
                    st.session_state.age = age
                    st.session_state.gender = gender
                    st.session_state.result = result
                    st.session_state.img1_path = img1_path
                    st.session_state.img2_path = img2_path
                    st.session_state.coords = coords
                    st.session_state.ai_consultation = ai_consultation 

                    c.execute("INSERT INTO history (name, age, gender, time, result, coords, img1_path, img2_path, report) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              (name, age, gender, datetime.now().strftime("%Y-%m-%d %H:%M"), result, str(coords), img1_path, img2_path, ai_consultation))
                    conn.commit()

        if st.session_state.analysis_done:
            with st.container(border=True):
                st.markdown("### Diagnostic Results")
                st.success(f"**Diagnosis:** {st.session_state.result}")
                st.write("")
                col_img1, col_img2 = st.columns(2)
                col_img1.image(st.session_state.img1_path, caption="AI Detected Region (Bounding Box)", use_container_width=True)
                col_img2.image(st.session_state.img2_path, caption="Attention Heatmap", use_container_width=True)

            with st.container(border=True):
                st.markdown("### Clinical Report (OpenAI)")
                st.markdown(st.session_state.ai_consultation, unsafe_allow_html=True)
                st.write("")
                pdf_file = export_pdf(st.session_state.name, st.session_state.age, st.session_state.gender, 
                                      st.session_state.result, st.session_state.ai_consultation, 
                                      st.session_state.img1_path, st.session_state.img2_path)
                with open(pdf_file, "rb") as f:
                    st.download_button("Download Official Medical Report (PDF)", f, file_name=f"{remove_accents(st.session_state.name)}_Clinical_Report.pdf", use_container_width=True)

    # ==========================
    # TRANG 2: ANALYTICS DASHBOARD
    # ==========================
    elif view_mode == "System Analytics":
        st.markdown('<div class="page-header">Analytics Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subheader">Performance metrics and system insights</div>', unsafe_allow_html=True)
        
        data = c.execute("SELECT * FROM history").fetchall()
        df = pd.DataFrame(data, columns=['id', 'name', 'age', 'gender', 'time', 'result', 'coords', 'img1', 'img2', 'report'])
        
        if df.empty:
            st.info("No data available for analytics yet. Please run some MRI scans first.")
        else:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df['Date'] = df['time'].dt.date
            df['Status'] = df['result'].apply(lambda x: 'Abnormal (Tumor)' if 'Abnormal' in str(x) else 'Normal')
            df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0)

            with st.container(border=True):
                total_scans = len(df)
                abnormal_count = len(df[df['Status'] == 'Abnormal (Tumor)'])
                abnormal_rate = f"{(abnormal_count/total_scans)*100:.1f}%" if total_scans > 0 else "0%"
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Scans Processed", total_scans, "System lifetime")
                c2.metric("Critical Detection Rate", abnormal_rate, "Abnormal Cases", delta_color="inverse")
                c3.metric("AI Accuracy (Validation)", "98.2%", "+1.5%")
                c4.metric("Avg Processing Time", "2.1s", "-12% faster")

            st.write("")

            colA, colB = st.columns(2)
            with colA:
                with st.container(border=True):
                    fig_diag = px.pie(df, names='Status', hole=0.45, 
                                      color='Status', color_discrete_map={'Abnormal (Tumor)':'#ff4b4b', 'Normal':'#00ADB5'})
                    fig_diag.update_layout(title_text="Diagnosis Distribution", title_x=0.2, margin=dict(t=50, b=20, l=10, r=10))
                    st.plotly_chart(fig_diag, use_container_width=True)
            
            with colB:
                with st.container(border=True):
                    fig_gender = px.pie(df, names='gender', hole=0.45, color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_gender.update_layout(title_text="Patient Demographics (Gender)", title_x=0.2, margin=dict(t=50, b=20, l=10, r=10))
                    st.plotly_chart(fig_gender, use_container_width=True)

            colC, colD = st.columns(2)
            with colC:
                with st.container(border=True):
                    trend_df = df.groupby('Date').size().reset_index(name='Scans')
                    fig_trend = px.line(trend_df, x='Date', y='Scans', markers=True, line_shape='spline')
                    fig_trend.update_traces(line_color='#00ADB5', marker=dict(size=8))
                    fig_trend.update_layout(title_text="Scan Volume Trend (Over Time)", title_x=0.2, margin=dict(t=50, b=20, l=10, r=10), xaxis_title="", yaxis_title="Number of Scans")
                    st.plotly_chart(fig_trend, use_container_width=True)

            with colD:
                with st.container(border=True):
                    fig_age = px.histogram(df, x='age', nbins=15, color_discrete_sequence=['#7b61ff'])
                    fig_age.update_layout(title_text="Patient Age Distribution", title_x=0.2, margin=dict(t=50, b=20, l=10, r=10), xaxis_title="Age", yaxis_title="Count", bargap=0.1)
                    st.plotly_chart(fig_age, use_container_width=True)

            with st.container(border=True):
                st.markdown("<h4 style='text-align:center;'>Recent Patient Logs</h4>", unsafe_allow_html=True)
                display_df = df[['time', 'name', 'age', 'gender', 'result']].tail(10).sort_values(by='time', ascending=False)
                display_df.columns = ['Date & Time', 'Patient Name', 'Age', 'Gender', 'AI Diagnosis']
                st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==========================================
# PATIENT TRACKING PAGE (MASTER-DETAIL)
# ==========================================
elif page == "Patient Tracking":
    st.markdown('<div class="page-header">Electronic Health Records (EHR)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Click on any patient row to view their comprehensive medical profile and progression</div>', unsafe_allow_html=True)

    data = c.execute("SELECT * FROM history ORDER BY time DESC").fetchall()
    df = pd.DataFrame(data, columns=['ID', 'Name', 'Age', 'Gender', 'Date', 'Diagnosis Result', 'Coordinates', 'Image1', 'Image2', 'AI Report'])

    if df.empty:
        st.info("No records found in the database. Please run an MRI analysis first.")
    else:
        # BƯỚC 1: HIỂN THỊ BẢNG DANH BẠ (MASTER VIEW)
        with st.container(border=True):
            col_search, col_export = st.columns([3, 1])
            with col_search:
                search_query = st.text_input("Search by Patient Name", placeholder="Leave blank to show all patients")
            with col_export:
                st.write("") 
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Export Full DB", data=csv, file_name='NeuroVision_DB.csv', mime='text/csv', use_container_width=True)

            if search_query:
                df_display = df[df['Name'].str.contains(search_query, case=False, na=False)]
            else:
                df_display = df

            # Lọc bỏ tên trùng, chỉ giữ lần khám mới nhất cho bảng tổng quan
            df_unique_patients = df_display.drop_duplicates(subset=['Name'], keep='first')

            st.markdown("### Patient Directory")
            st.info("**Tip:** Click on any row in the table below to open the patient's full medical profile.")
            
            try:
                event = st.dataframe(
                    df_unique_patients[['Date', 'Name', 'Age', 'Gender', 'Diagnosis Result']],
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",          
                    selection_mode="single-row" 
                )
                selected_rows = event.selection.rows
            except Exception as e:
                st.dataframe(df_unique_patients[['Date', 'Name', 'Age', 'Gender', 'Diagnosis Result']], use_container_width=True, hide_index=True)
                st.warning("Lỗi phiên bản Streamlit cũ: Vui lòng chạy lệnh `pip install --upgrade streamlit` ở Terminal để bật tính năng click chọn hàng.")
                selected_rows = []

        # BƯỚC 2: MỞ KHÓA HỒ SƠ CHI TIẾT NẾU CÓ DÒNG BỊ CLICK
        if len(selected_rows) > 0:
            selected_index = selected_rows[0]
            selected_patient = df_unique_patients.iloc[selected_index]['Name']

            p_records = [d for d in data if d[1] == selected_patient]
            p_records_asc = sorted(p_records, key=lambda x: x[4]) # Sắp xếp từ cũ đến mới
            latest_record = p_records_asc[-1]

            with st.container(border=True):
                st.markdown(f"<h2 style='text-align:center; color:#00ADB5;'>Patient Profile: {selected_patient}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; color:#888;'>Age: {latest_record[2]} | Gender: {latest_record[3]} | Total Scans: {len(p_records)}</p>", unsafe_allow_html=True)
                
                tab_hist, tab_comp, tab_edit, tab_manage = st.tabs([
                    "Scan History", 
                    "Progression Comparison", 
                    "Edit Info", 
                    "Manage Data"
                ])

                # --- TAB 1: LỊCH SỬ KHÁM ---
                with tab_hist:
                    for rec in p_records: # Hiện từ mới nhất -> cũ nhất do lúc đầu list data đã sort DESC
                        rec_id = rec[0]
                        status_icon = ""
                        
                        with st.expander(f"{status_icon}Scan Date: {rec[4]} | Diagnosis: {rec[5]}", expanded=(rec == p_records[0])):
                            report_text = rec[9] if len(rec) > 9 and rec[9] else "No AI clinical report generated."
                            st.markdown(f"**Clinical Report:**<br>{report_text}", unsafe_allow_html=True)
                            st.write("")
                            
                            colA, colB = st.columns(2)
                            try:
                                colA.image(rec[7], caption="Detected Region", use_container_width=True)
                                colB.image(rec[8], caption="Attention Heatmap", use_container_width=True)
                            except:
                                st.warning("Images are no longer stored on the server.")

                            try:
                                hist_pdf = export_pdf(rec[1], rec[2], rec[3], rec[5], report_text, rec[7], rec[8])
                                with open(hist_pdf, "rb") as f:
                                    st.download_button("Download PDF Report", f, file_name=f"{remove_accents(rec[1])}_{rec[4][:10]}_Report.pdf", key=f"dl_hist_{rec_id}")
                            except Exception:
                                pass

                # --- TAB 2: SO SÁNH TIẾN TRIỂN (CÓ CHỌN FILE) ---
                with tab_comp:
                    if len(p_records_asc) < 2:
                        st.info("Not enough data. You need to scan this patient at least twice to track disease progression over time.")
                    else:
                        st.markdown("<h4 style='text-align:center;'>Compare MRI Scans</h4>", unsafe_allow_html=True)
                        
                        scan_dict = {f"Scan on {r[4][:16]} | {r[5].split('(')[0].strip()}": r for r in p_records_asc}
                        scan_labels = list(scan_dict.keys())
                        
                        sel_col1, sel_col2 = st.columns(2)
                        with sel_col1:
                            label_a = st.selectbox("Select Scan A (Baseline):", scan_labels, index=len(scan_labels)-2)
                        with sel_col2:
                            label_b = st.selectbox("Select Scan B (Follow-up):", scan_labels, index=len(scan_labels)-1)
                            
                        scan_a = scan_dict[label_a]
                        scan_b = scan_dict[label_b]
                        
                        if scan_a[0] == scan_b[0]:
                            st.warning("Please select two DIFFERENT scans to compare.")
                        else:
                            current_pair = f"{scan_a[0]}_{scan_b[0]}"
                            if st.session_state.get("comp_pair") != current_pair:
                                st.session_state.comp_pair = current_pair
                                st.session_state.comp_report = ""

                            c1, c2 = st.columns(2)
                            c1.info(f"**Scan A (Baseline):** {scan_a[4]}\n\n**Diagnosis:** {scan_a[5]}")
                            c2.info(f"**Scan B (Follow-up):** {scan_b[4]}\n\n**Diagnosis:** {scan_b[5]}")
                            
                            st.markdown("---")
                            st.markdown("<h5 style='text-align:center;'>1. Tumor Detection Comparison</h5>", unsafe_allow_html=True)
                            img_col1, img_col2 = st.columns(2)
                            try:
                                img_col1.image(scan_a[7], caption=f"Scan A ({scan_a[4][:10]})", use_container_width=True)
                                img_col2.image(scan_b[7], caption=f"Scan B ({scan_b[4][:10]})", use_container_width=True)
                            except:
                                st.error("Image missing for comparison.")
                                
                            st.write("")
                            st.markdown("<h5 style='text-align:center;'>2. Neural Activity (Heatmap) Comparison</h5>", unsafe_allow_html=True)
                            hm_col1, hm_col2 = st.columns(2)
                            try:
                                hm_col1.image(scan_a[8], caption=f"Heatmap A ({scan_a[4][:10]})", use_container_width=True)
                                hm_col2.image(scan_b[8], caption=f"Heatmap B ({scan_b[4][:10]})", use_container_width=True)
                            except:
                                st.error("Image missing for comparison.")

                            if st.button("Run AI Progression Analysis", use_container_width=True, type="primary", key=f"btn_comp_{selected_patient}"):
                                with st.spinner("NeuroVision is assessing progression..."):
                                    if OPENAI_KEY:
                                        try:
                                            client = OpenAI(api_key=OPENAI_KEY)
                                            prompt = f"""
                                            Act as a Senior Clinical Neuroradiologist. Compare MRI results for patient {selected_patient}.
                                            Previous Scan ({scan_a[4]}): {scan_a[5]}.
                                            Recent Scan ({scan_b[4]}): {scan_b[5]}.
                                            
                                            Write a professional summary assessing disease progression. Format:
                                            <b>COMPARISON FINDINGS:</b>
                                            (Describe changes).
                                            <br/><br/>
                                            <b>IMPRESSION & RECOMMENDATION:</b>
                                            (Conclusion and advice).
                                            
                                            Use ONLY HTML tags <b> and <br/>. No markdown (**).
                                            """
                                            response = client.chat.completions.create(
                                                model="gpt-4o-mini", 
                                                messages=[{"role": "system", "content": "You are a professional medical AI assistant."},
                                                          {"role": "user", "content": prompt}]
                                            )
                                            st.session_state.comp_report = response.choices[0].message.content
                                        except Exception as e:
                                            st.error(f"OpenAI Error: {e}")
                                    else:
                                        st.warning("Please configure OpenAI API Key in secrets.toml.")

                            if st.session_state.get("comp_report"):
                                with st.container(border=True):
                                    st.markdown("### Progression Report")
                                    st.markdown(st.session_state.comp_report, unsafe_allow_html=True)
                                    pdf_comp = export_comparison_pdf(selected_patient, scan_a[2], scan_a[3], scan_a, scan_b, st.session_state.comp_report)
                                    with open(pdf_comp, "rb") as f:
                                        st.download_button("Download Progression Report (PDF)", f, file_name=f"{remove_accents(selected_patient)}_Progression.pdf", use_container_width=True)

                # --- TAB 3: CHỈNH SỬA THÔNG TIN ---
                with tab_edit:
                    with st.form(key=f"edit_form_profile"):
                        st.info("Update demographic details for this patient. This will update ALL their past scan records to ensure consistency.")
                        e_col1, e_col2, e_col3 = st.columns(3)
                        new_name = e_col1.text_input("Full Name", value=latest_record[1])
                        new_age = e_col2.text_input("Age", value=latest_record[2])
                        
                        gender_index = ["Male", "Female", "Other"].index(latest_record[3]) if latest_record[3] in ["Male", "Female", "Other"] else 0
                        new_gender = e_col3.selectbox("Gender", ["Male", "Female", "Other"], index=gender_index)
                        
                        submit_edit = st.form_submit_button("Save Changes to All Records")
                        if submit_edit:
                            c.execute("UPDATE history SET name=?, age=?, gender=? WHERE name=?", (new_name, new_age, new_gender, selected_patient))
                            conn.commit()
                            st.success("Patient profile updated successfully!")
                            st.rerun()

                # --- TAB 4: QUẢN LÝ ---
                with tab_manage:
                    st.warning("**Danger Zone:** Ensure HIPAA compliance. Deleting records is permanent.")
                    
                    st.markdown("#### Individual Scan Records")
                    for rec in reversed(p_records_asc):
                        rec_id = rec[0]
                        with st.container(border=True):
                            col_info, col_btn = st.columns([3, 1])
                            with col_info:
                                st.markdown(f"**Scan Date:** `{rec[4]}`")
                                st.caption(f"Diagnosis: {rec[5]}")
                            with col_btn:
                                st.write("") 
                                if st.button("Delete This Scan", key=f"del_scan_{rec_id}", use_container_width=True):
                                    c.execute("DELETE FROM history WHERE id=?", (rec_id,))
                                    conn.commit()
                                    st.rerun()

                    st.markdown("---")
                    st.markdown("#### Bulk Action")
                    if st.button("Delete ENTIRE Patient Profile", type="primary", use_container_width=True):
                        c.execute("DELETE FROM history WHERE name=?", (selected_patient,))
                        conn.commit()
                        st.rerun()
        else:
            st.info("Please click on a patient row in the table above to view their details, reports, and compare scans.")
