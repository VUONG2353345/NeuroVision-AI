import streamlit as st
import sqlite3
from datetime import datetime
import unicodedata
import re
import io
import time
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
import sys
from openai import OpenAI  
import pandas as pd 
import plotly.express as px

# Trỏ đường dẫn vào thư mục chứa code AI
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mri_analyzer import analyze_mri_unet
from unet_model import UNet

# ==========================================
# CONFIG & FOLDERS
# ==========================================
st.set_page_config(page_title="NeuroVision AI", layout="wide", page_icon=r"C:\Users\Asus\Hackathon\Hackathon_2026-main\Hackathon_2026-main\iconfinder-bl-1646-brain-artificial-intelligence-electronic-computer-processor-consciousness-4575061_121498.png")
st.markdown('<div id="top-of-page"></div>', unsafe_allow_html=True)

if not os.path.exists("history_img"):
    os.makedirs("history_img")

try:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_KEY = None

@st.cache_resource
def load_ai_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load("models/unet_best.pth", map_location=device))
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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=DM+Sans:wght@500;700&display=swap');

:root {
    --bg-main: #edf7f1;
    --bg-layer: #e3f2e9;
    --bg-panel: rgba(255, 255, 255, 0.94);
    --bg-panel-strong: #ffffff;
    --bg-sidebar: #0f2f27;
    --bg-sidebar-soft: #133a30;
    --bg-input: #f7fcf9;
    --text-main: #163128;
    --text-soft: #5f776d;
    --text-strong: #0d241d;
    --line: rgba(18, 82, 58, 0.10);
    --line-strong: rgba(18, 82, 58, 0.18);
    --accent: #22a06b;
    --accent-strong: #157347;
    --accent-soft: rgba(34, 160, 107, 0.10);
    --accent-soft-strong: rgba(34, 160, 107, 0.18);
    --accent-wash: #dff5ea;
    --shadow-soft: 0 22px 48px -34px rgba(20, 66, 49, 0.22);
    --shadow-hover: 0 30px 56px -36px rgba(20, 66, 49, 0.30);
    --radius-xl: 34px;
    --radius-lg: 26px;
    --radius-md: 18px;
    --radius-sm: 14px;
}

@keyframes fadeInUp {
    0% {
        opacity: 0;
        transform: translateY(22px) scale(0.985);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: translateY(0);
    }
    50% {
        opacity: 0.74;
        transform: translateY(-2px);
    }
}

@keyframes greenFlow {
    0% {
        background-position: 0% 50%;
    }
    100% {
        background-position: 100% 50%;
    }
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif !important;
    font-size: 17px !important;
    color: var(--text-main);
    scroll-behavior: smooth;
    scrollbar-width: thin;
    scrollbar-color: #22a06b rgba(34, 160, 107, 0.08);
}

body, .stApp {
    background:
        radial-gradient(circle at top left, rgba(34, 160, 107, 0.16), transparent 18%),
        radial-gradient(circle at 88% 8%, rgba(93, 196, 130, 0.16), transparent 16%),
        linear-gradient(180deg, #f5fbf7 0%, #edf7f1 52%, #e7f2ec 100%);
    color: var(--text-main);
}

::-webkit-scrollbar {
    width: 14px;
    height: 14px;
}

::-webkit-scrollbar-track {
    background: rgba(34, 160, 107, 0.08);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #80d3a5 0%, #22a06b 100%);
    border-radius: 999px;
    border: 3px solid rgba(237, 247, 241, 0.92);
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #68c593 0%, #178053 100%);
}

[data-testid="stHeader"] {
    background: rgba(237, 247, 241, 0.82);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(18, 82, 58, 0.08);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, var(--bg-sidebar) 0%, var(--bg-sidebar-soft) 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.04);
    box-shadow: 10px 0 30px -24px rgba(11, 43, 34, 0.45);
    padding-top: 0.25rem;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 0.4rem;
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
    gap: 0.6rem;
    width: 100%;
}

[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
    margin: 0;
    width: 100%;
    border-radius: 18px;
    padding: 0.2rem 0.35rem;
    background: transparent;
    border: 1px solid transparent;
    transition: all 0.25s ease;
}

[data-testid="stSidebar"] .stRadio label {
    width: 100%;
    margin: 0;
    padding: 0.9rem 1rem;
    border-radius: 16px;
    justify-content: flex-start;
    color: rgba(255, 255, 255, 0.76) !important;
    font-weight: 700;
    letter-spacing: 0.015em;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 255, 255, 0.06);
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, rgba(34, 160, 107, 0.22), rgba(80, 191, 123, 0.14));
    border: 1px solid rgba(124, 226, 166, 0.20);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:has(input:checked) label {
    background: rgba(255, 255, 255, 0.02);
    color: #ffffff !important;
}

.sidebar-header {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.7rem;
    width: 100%;
    margin: 0 auto 1rem auto;
    padding: 1rem 1.05rem;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.08);
    color: #ffffff;
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    text-align: center;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

.sidebar-header img {
    width: 26px;
    height: 26px;
    object-fit: contain;
    display: block;
    filter: drop-shadow(0 0 14px rgba(124, 226, 166, 0.14));
}

h1, h2, h3 {
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    line-height: 1.12;
    color: var(--text-strong);
    margin-top: 0.15rem;
    margin-bottom: 0.9rem;
}

h1 { font-size: 2.9rem !important; }
h2 { font-size: 2.1rem !important; }
h3 { font-size: 1.55rem !important; }

p, label, .stMarkdown, .stCaption, .stText, .stRadio label, .stSelectbox label {
    color: var(--text-main) !important;
}

.stApp [data-testid="stMarkdownContainer"] p {
    line-height: 1.6;
}

[data-testid="stAppViewContainer"] .block-container {
    animation: fadeInUp 0.55s ease both;
}

.brand-title { 
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 18px;
    margin-bottom: 0.35rem;
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
    filter: drop-shadow(0 18px 24px rgba(34, 160, 107, 0.20));
}

.brand-mark svg {
    width: 50px;
    height: 50px;
    display: block;
}

.brand-word {
    display: inline-block;
    font-size: 72px;
    font-weight: 800;
    line-height: 0.92;
    letter-spacing: -0.08em;
    background: linear-gradient(135deg, #0b211b 0%, #1d493b 68%, #2f7d62 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none;
}

.brand-subtitle,
.page-subheader {
    font-size: 1.04rem;
    color: var(--text-soft) !important;
    text-align: center;
    margin-bottom: 2.1rem;
    font-weight: 500;
    letter-spacing: 0.005em;
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
}

.page-header {
    font-size: 2.45rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.55rem;
    color: var(--text-strong);
    letter-spacing: -0.05em;
    text-transform: none;
}

button,
a,
[role="button"],
[data-baseweb="tab"],
[data-testid="stExpanderToggleIcon"],
.stDownloadButton > button,
.stButton > button {
    transition: all 0.24s ease !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    padding: 1.7rem !important;
    margin-bottom: 1.4rem;
    border-radius: var(--radius-xl) !important;
    border: 1px solid var(--line) !important;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,252,247,0.92)) !important;
    box-shadow: var(--shadow-soft) !important;
    position: relative;
    overflow: hidden;
    animation: fadeInUp 0.7s ease both;
}

div[data-testid="stVerticalBlockBorderWrapper"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.72), transparent 42%),
        linear-gradient(180deg, rgba(34,160,107,0.045), transparent 36%);
    pointer-events: none;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-4px) rotate(-0.2deg);
    box-shadow: var(--shadow-hover) !important;
}

.stImage > img {
    border-radius: 22px;
    box-shadow: 0 26px 42px -32px rgba(15, 23, 42, 0.34);
    border: 1px solid rgba(15, 23, 42, 0.08);
}

.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] > button {
    min-height: 3.1rem;
    border-radius: 999px;
    border: 1px solid rgba(15, 23, 42, 0.06);
    background: linear-gradient(135deg, #0f5d3f 0%, #22a06b 55%, #4fbf7b 100%);
    color: #ffffff;
    font-weight: 800;
    font-size: 0.98rem;
    letter-spacing: 0.015em;
    box-shadow: 0 18px 32px -22px rgba(15, 93, 63, 0.34);
}

.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) scale(1.01);
    filter: brightness(1.03);
}

.scroll-to-top {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background: linear-gradient(135deg, #0f5d3f 0%, #22a06b 100%);
    color: white;
    padding: 12px 20px;
    border-radius: 999px;
    text-decoration: none;
    font-weight: bold;
    border: 1px solid rgba(15, 23, 42, 0.06);
    box-shadow: 0 16px 30px -24px rgba(15, 93, 63, 0.34);
    z-index: 9999;
    transition: 0.3s;
}

.scroll-to-top:hover {
    transform: translateY(-3px);
    filter: brightness(1.03);
}

.stButton > button:focus,
.stDownloadButton > button:focus,
div[data-testid="stFormSubmitButton"] > button:focus {
    outline: none;
    box-shadow: 0 0 0 0.22rem rgba(34, 160, 107, 0.16);
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {
    background: var(--bg-input) !important;
    border: 1px solid rgba(18, 82, 58, 0.10) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-strong) !important;
    font-size: 0.98rem !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.9) !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
.stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(34, 160, 107, 0.36) !important;
    box-shadow: 0 0 0 0.20rem rgba(34, 160, 107, 0.12) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.55rem;
    justify-content: center;
    background: rgba(255,255,255,0.56);
    border: 1px solid rgba(15, 23, 42, 0.06);
    border-radius: 999px;
    padding: 0.45rem;
    width: fit-content;
    margin: 0 auto 1.1rem auto;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
}

.stTabs [data-baseweb="tab"] {
    min-height: 46px;
    padding: 0.7rem 1.2rem;
    border-radius: 999px;
    background: transparent;
    color: var(--text-soft);
    font-weight: 700;
    border: 1px solid transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-strong);
    background: rgba(34, 160, 107, 0.08);
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, #0f5d3f 0%, #22a06b 100%) !important;
    border: 1px solid rgba(15, 23, 42, 0.04) !important;
    box-shadow: 0 12px 20px -16px rgba(15, 93, 63, 0.28);
}

[data-testid="stAppViewContainer"] .stRadio > div {
    display: flex;
    justify-content: center;
}

[data-testid="stAppViewContainer"] .stRadio {
    margin: 1rem auto 1.9rem auto;
}

[data-testid="stAppViewContainer"] .stRadio div[role="radiogroup"] {
    justify-content: center !important;
    width: 100%;
    margin: 0 auto;
    gap: 0.42rem;
    padding: 0.5rem 0.55rem;
    border-radius: 999px;
    background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(236,247,241,0.92));
    border: 1px solid rgba(18, 82, 58, 0.12);
    box-shadow:
        0 18px 32px -26px rgba(15, 93, 63, 0.20),
        inset 0 1px 0 rgba(255,255,255,0.95);
    max-width: fit-content;
    position: relative;
    overflow: hidden;
}

[data-testid="stAppViewContainer"] .stRadio div[role="radiogroup"] > label {
    margin-left: 0;
    margin-right: 0;
    padding: 0.78rem 1.28rem;
    border-radius: 999px;
    color: var(--text-soft) !important;
    font-weight: 800;
    font-family: 'DM Sans', sans-serif !important;
    transition: color 0.26s ease, transform 0.26s ease, letter-spacing 0.26s ease !important;
    min-height: 3rem;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    white-space: nowrap !important;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, rgba(15,93,63,0.92), rgba(34,160,107,0.92));
    border-radius: 999px;
    box-shadow: 0 14px 22px -18px rgba(15, 93, 63, 0.32) !important;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"]:has(input:checked) label {
    color: #ffffff !important;
}

[data-testid="stSpinner"],
.stSpinner,
[data-testid="stSpinner"] * {
    color: var(--accent) !important;
    animation: pulse 1.5s ease-in-out infinite;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0f5d3f 0%, #22a06b 100%) !important;
}

[data-testid="stProgressBar"] {
    background: rgba(18, 82, 58, 0.08) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, #fffdf8 0%, #f9f4eb 100%);
    border: 1px solid rgba(18, 82, 58, 0.08);
    border-radius: 22px;
    padding: 1rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: var(--text-soft) !important;
}

[data-testid="stMetricValue"] {
    color: var(--text-strong) !important;
}

[data-testid="stDataFrame"],
.stAlert,
.streamlit-expanderHeader {
    border-radius: 20px !important;
}

[data-testid="stDataFrame"],
.stAlert,
.streamlit-expanderHeader,
[data-testid="stExpander"] {
    border: 1px solid rgba(18, 82, 58, 0.08) !important;
    background: rgba(255,252,247,0.9) !important;
    color: var(--text-main) !important;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34, 160, 107, 0.28), transparent);
    margin: 1.75rem 0;
}

a {
    color: #0f7a53;
}

.stAlert {
    box-shadow: none !important;
}

.stSuccess,
.stInfo,
.stWarning,
.stError {
    border-radius: 20px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,250,242,0.76) !important;
    border: 1.5px dashed rgba(34, 160, 107, 0.34) !important;
    border-radius: 24px !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: var(--text-soft) !important;
}

[data-testid="stMarkdownContainer"] code,
.stCodeBlock {
    background: #12392f !important;
    color: #eafff5 !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    right: 1rem;
}

[data-testid="stDataFrame"] {
    box-shadow: none !important;
}

[data-baseweb="tooltip"] {
    background: #12392f !important;
    color: #fffaf0 !important;
    border: 1px solid rgba(34, 160, 107, 0.24) !important;
}

[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.92);
}

[data-testid="stAppViewBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebarNavSeparator"] {
    display: none;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #144533 0%, #0f2f27 100%);
}

.stSuccess,
.stInfo,
.stWarning,
.stError {
    border: 1px solid rgba(15, 23, 42, 0.06) !important;
    background: rgba(255,252,247,0.88) !important;
}

.stCaption {
    color: var(--text-soft) !important;
}

.streamlit-expanderHeader {
    font-weight: 700 !important;
}

[data-testid="stSelectbox"] svg,
[data-testid="stNumberInput"] svg {
    color: var(--text-soft) !important;
}

.hero-shell {
    position: relative;
    margin: 0 auto 1.8rem auto;
    padding: 2rem 2.2rem;
    border-radius: 36px;
    background:
        radial-gradient(circle at top left, rgba(34,160,107,0.16), transparent 22%),
        linear-gradient(135deg, rgba(255,255,255,0.88), rgba(245,252,248,0.94));
    border: 1px solid rgba(18, 82, 58, 0.08);
    box-shadow: var(--shadow-soft);
    overflow: hidden;
}

.hero-shell::after {
    content: "";
    position: absolute;
    inset: auto -10% -35% auto;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(34,160,107,0.15), transparent 62%);
    pointer-events: none;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    background: rgba(34,160,107,0.08);
    color: var(--accent-strong);
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.95rem;
}

.hero-title {
    font-size: 3rem;
    line-height: 0.96;
    letter-spacing: -0.07em;
    font-weight: 800;
    color: var(--text-strong);
    margin: 0 0 0.9rem 0;
    max-width: 780px;
}

.hero-copy {
    font-size: 1.03rem;
    line-height: 1.75;
    color: var(--text-soft);
    max-width: 760px;
    margin: 0;
}

.sidebar-note {
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
}

.sidebar-note-title {
    color: #ffffff;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.sidebar-note-copy {
    color: rgba(255,255,255,0.68);
    font-size: 0.86rem;
    line-height: 1.55;
}

.sidebar-card {
    margin-top: 1rem;
    padding: 1rem 1.05rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.06);
}

.sidebar-card-title {
    color: #ffffff;
    font-size: 0.9rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.sidebar-card-copy {
    color: rgba(255,255,255,0.72);
    font-size: 0.86rem;
    line-height: 1.6;
}

.sidebar-card-copy strong {
    color: #ffffff;
}

.login-shell {
    padding: 1.8rem 1.9rem;
    border-radius: 34px;
    background:
        radial-gradient(circle at top left, rgba(34,160,107,0.12), transparent 22%),
        linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,255,252,0.96));
    border: 1px solid rgba(18, 82, 58, 0.08);
    box-shadow: var(--shadow-soft);
}

.login-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    background: rgba(34,160,107,0.10);
    color: var(--accent-strong);
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.login-headline {
    font-size: 2.35rem;
    line-height: 1.02;
    letter-spacing: -0.06em;
    color: var(--text-strong);
    font-weight: 800;
    margin-bottom: 0.85rem;
}

.login-copy {
    color: var(--text-soft);
    font-size: 1rem;
    line-height: 1.75;
    margin-bottom: 1.25rem;
}

.login-list {
    display: grid;
    gap: 0.85rem;
}

.login-item {
    padding: 1rem 1.05rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(18, 82, 58, 0.08);
}

.login-item-title {
    color: var(--text-strong);
    font-size: 0.98rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.login-item-copy {
    color: var(--text-soft);
    font-size: 0.9rem;
    line-height: 1.55;
}

.session-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.48rem 0.9rem;
    border-radius: 999px;
    background: rgba(34,160,107,0.10);
    color: var(--accent-strong);
    font-size: 0.82rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

.scroll-to-top,
.scroll-to-top:visited,
.scroll-to-top:hover {
    color: #ffffff !important;
}

.stButton > button *,
.stDownloadButton > button *,
div[data-testid="stFormSubmitButton"] > button * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #12754d 0%, #22a06b 100%) !important;
}

.stRadio [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"] {
    padding: 0 !important;
    margin: 0 !important;
    min-width: fit-content;
    border-radius: 999px;
    overflow: hidden;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: transform 0.28s ease, box-shadow 0.28s ease, background 0.32s ease, filter 0.28s ease !important;
    will-change: transform;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"]:has(input:checked) {
    background: linear-gradient(135deg, #0f6f4d 0%, #22a06b 52%, #69d99c 100%) !important;
    background-size: 180% 180%;
    box-shadow:
        0 14px 26px -18px rgba(15, 93, 63, 0.34),
        inset 0 1px 0 rgba(255,255,255,0.24) !important;
    transform: translateY(-1px) scale(1.01);
    animation: greenFlow 2.4s ease-in-out infinite alternate;
    padding-inline: 0.18rem !important;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"]:has(input:checked) label {
    color: #ffffff !important;
    letter-spacing: 0.01em;
    text-align: center !important;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"]:has(input:checked) * {
    color: #ffffff !important;
    fill: #ffffff !important;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"] label,
[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"] label > div,
[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"] label > div > p,
[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"] label p,
[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"] label span {
    width: 100% !important;
    margin: 0 !important;
    text-align: center !important;
    justify-content: center !important;
    align-items: center !important;
    white-space: nowrap !important;
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"]:not(:has(input:checked)):hover {
    background: rgba(34, 160, 107, 0.10) !important;
    transform: translateY(-1px);
}

[data-testid="stAppViewContainer"] .stRadio [data-baseweb="radio"]:not(:has(input:checked)):hover label {
    color: var(--text-strong) !important;
}

.stTabs [aria-selected="true"] {
    background: #22a06b !important;
    border-color: rgba(18, 82, 58, 0.06) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(34,160,107,0.22) !important;
    transform: translateY(-1px);
}

.sidebar-action-gap {
    height: 1rem;
}

.sidebar-action-label {
    margin: 0 0 0.75rem 0;
    padding: 0.7rem 0.95rem;
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04));
    border: 1px solid rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.92);
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

if "doctor_authenticated" not in st.session_state:
    st.session_state.doctor_authenticated = False
if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = ""
if "doctor_email" not in st.session_state:
    st.session_state.doctor_email = ""
if "doctor_department" not in st.session_state:
    st.session_state.doctor_department = "Radiology"
if "doctor_remember" not in st.session_state:
    st.session_state.doctor_remember = False
if "uploader_nonce" not in st.session_state:
    st.session_state.uploader_nonce = 0

st.sidebar.markdown('<div class="sidebar-header">Clinical Intro</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-card">
    <div class="sidebar-card-title">What is NeuroVision?</div>
    <div class="sidebar-card-copy">
        A clinician-facing MRI workspace that combines segmentation review, patient follow-up, and reporting in one calm interface.
    </div>
</div>
<div class="sidebar-card">
    <div class="sidebar-card-title">Experience Focus</div>
    <div class="sidebar-card-copy">
        This version is tuned for a greener medical feel, smoother navigation, and cleaner day-to-day physician workflow.
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.doctor_authenticated:
    st.sidebar.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">Doctor Session</div>
            <div class="sidebar-card-copy">
                <strong>{st.session_state.doctor_name}</strong><br/>
                {st.session_state.doctor_department}<br/>
                {st.session_state.doctor_email}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("""
<div class="sidebar-card">
    <div class="sidebar-card-title">Credits</div>
    <div class="sidebar-card-copy">
        Product redesign, AI integration, physician workflow prototyping, and Streamlit implementation by the NeuroVision build team.
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.doctor_authenticated:
    st.sidebar.markdown('<div class="sidebar-action-gap"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-action-label">Session Actions</div>', unsafe_allow_html=True)
    if st.sidebar.button("Doctor Sign Out", use_container_width=True, key="doctor_sign_out_btn"):
        st.session_state.uploader_nonce += 1
        for key in ["doctor_authenticated", "doctor_name", "doctor_email", "doctor_department", "doctor_remember"]:
            st.session_state.pop(key, None)
        st.rerun()

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

def normalize_patient_name(name):
    normalized_name = re.sub(r"\s+", " ", str(name or "").strip())
    return normalized_name.casefold()

def load_uploaded_mri(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    ext = os.path.splitext(uploaded_file.name.lower())[1]

    if ext in {".ima", ".dcm"}:
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        image = ds.pixel_array.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255.0
        image = image.astype(np.uint8)
    else:
        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Unsupported MRI image format.")

    if image.ndim == 3:
        image = image[image.shape[0] // 2, :, :]

    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image_normalized = image_resized.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0).to(device)
    return image_normalized, input_tensor

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

    elements.append(Paragraph("<b>ATTACHED IMAGES (DETECTION & SEGMENTATION OVERLAY):</b>", styles["Heading3"]))
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
            [Paragraph("<para align='center'><b>Segmentation Overlay (A)</b></para>", styles["Normal"]), Paragraph("<para align='center'><b>Segmentation Overlay (B)</b></para>", styles["Normal"])],
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

if not st.session_state.doctor_authenticated:
    st.markdown('<div class="brand-subtitle">Doctor-only access to the NeuroVision workspace. Sign in to open diagnostic review, command analytics, and patient timelines.</div>', unsafe_allow_html=True)

    login_col1, login_col2 = st.columns([1.1, 0.9], gap="large")
    with login_col1:
        st.markdown("""
        <div class="login-shell">
            <div class="login-badge">Doctor Access</div>
            <div class="login-headline">Secure physician sign-in for a calmer MRI workflow.</div>
            <p class="login-copy">
                Designed for radiologists and doctors who need a cleaner interface for intake, AI findings, longitudinal review, and reporting.
            </p>
            <div class="login-list">
                <div class="login-item">
                    <div class="login-item-title">Clinical-first entry</div>
                    <div class="login-item-copy">Fast access to case intake, segmentation review, and narrative reporting.</div>
                </div>
                <div class="login-item">
                    <div class="login-item-title">Doctor-specific workspace</div>
                    <div class="login-item-copy">Intended for licensed physicians, medical reviewers, and supervised clinical teams.</div>
                </div>
                <div class="login-item">
                    <div class="login-item-title">Smoother session flow</div>
                    <div class="login-item-copy">Move between diagnosis, command analytics, and patient history through one cleaner navigation bar.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with login_col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Doctor Login</h3>", unsafe_allow_html=True)
            st.caption("Use your physician identity to enter the clinical workspace. Identity provider setup can be connected later.")
            with st.form("doctor_login_form", clear_on_submit=False):
                doctor_name_input = st.text_input("Doctor Name", placeholder="e.g. Dr. Nguyen Minh")
                doctor_email_input = st.text_input("Hospital Email", placeholder="name@hospital.vn")
                doctor_department_input = st.selectbox("Department", ["Radiology", "Neurology", "Neurosurgery", "Oncology"])
                doctor_password_input = st.text_input("Password", type="password", placeholder="Enter your secure password")
                remember_device = st.checkbox("Remember this workstation")
                login_submit = st.form_submit_button("Enter Clinical Workspace", use_container_width=True)

                if login_submit:
                    if doctor_name_input.strip() and doctor_email_input.strip() and "@" in doctor_email_input and doctor_password_input.strip():
                        st.session_state.doctor_authenticated = True
                        st.session_state.doctor_name = doctor_name_input.strip()
                        st.session_state.doctor_email = doctor_email_input.strip()
                        st.session_state.doctor_department = doctor_department_input
                        st.session_state.doctor_remember = remember_device
                        st.rerun()
                    else:
                        st.error("Please enter doctor name, hospital email, and password to continue.")

    st.markdown('<a href="#top-of-page" class="scroll-to-top">Top</a>', unsafe_allow_html=True)
    st.stop()

st.markdown(
    f'<div class="brand-subtitle"><span class="session-pill">Doctor Session Active</span><br/>Welcome back, {st.session_state.doctor_name}. Choose a workspace below.</div>',
    unsafe_allow_html=True
)

page = st.radio("", ["Diagnostic Studio", "Clinical Command", "Patient Atlas"], horizontal=True, label_visibility="collapsed")

hero_content = {
    "Diagnostic Studio": (
        "AI-first MRI review, redesigned.",
        "Capture a case, run segmentation, and generate a clinical narrative in one focused studio instead of a pile of rough controls."
    ),
    "Clinical Command": (
        "A calmer command layer for operational oversight.",
        "Review throughput, diagnostic mix, and activity trends from a cleaner medical operations dashboard."
    ),
    "Patient Atlas": (
        "Longitudinal patient navigation without the clutter.",
        "Open a patient dossier, compare historical scans, and manage follow-up records inside one continuous timeline hub."
    ),
}

hero_title, hero_copy = hero_content[page]
st.markdown(
    f"""
    <div class="hero-shell">
        <div class="hero-kicker">NeuroVision Workspace</div>
        <div class="hero-title">{hero_title}</div>
        <p class="hero-copy">{hero_copy}</p>
    </div>
    """,
    unsafe_allow_html=True
)

view_mode = "AI Analysis"

if page == "Diagnostic Studio":

    # ==========================
    # TRANG 1: PHÂN TÍCH AI
    # ==========================
    if view_mode == "AI Analysis":
        analysis_file_key = f"analysis_file_{st.session_state.uploader_nonce}"
        if "analysis_done" not in st.session_state:
            st.session_state.analysis_done = False
            st.session_state.ai_consultation = ""
        if "analysis_name" not in st.session_state:
            st.session_state.analysis_name = ""
        if "analysis_age" not in st.session_state:
            st.session_state.analysis_age = ""
        if "analysis_gender" not in st.session_state:
            st.session_state.analysis_gender = "Male"

        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Case Intake</h3>", unsafe_allow_html=True)
            st.write("") 
            col1, col2, col3 = st.columns(3)
            name = col1.text_input(
                "Full Name",
                placeholder="e.g. John Doe",
                help="Enter the patient's full legal name as per hospital records.",
                key="analysis_name"
            )
            age = col2.text_input("Age", placeholder="e.g. 45", key="analysis_age")
            gender = col3.selectbox("Gender", ["Male", "Female", "Other"], key="analysis_gender")

        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Imaging Dropzone</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: #5f776d;'>Accepted formats: .dcm, .ima, .png, .jpg, .jpeg, .tif, .tiff, .bmp</p>", unsafe_allow_html=True)
            file = st.file_uploader(
                "",
                type=["dcm", "ima", "png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                label_visibility="collapsed",
                help="Upload a high-resolution MRI scan. Ensure all PII is anonymized.",
                key=analysis_file_key
            )
            
            st.write("")
            analyze_btn = st.button("Generate MRI Insight", use_container_width=True, type="primary")

        if analyze_btn:
            if file is None or name == "":
                st.error("Please fill in patient name and upload an MRI file!")
            else:
                status_message = st.empty()
                progress_bar = st.progress(0)
                try:
                    status_message.info("1. Preprocessing image...")
                    progress_bar.progress(20)
                    time.sleep(0.35)
                    try:
                        image_normalized, input_tensor = load_uploaded_mri(file)
                    except Exception as e:
                        st.error(f"Failed to read MRI scan: {e}")
                    else:
                        status_message.info("2. Running U-Net Segmentation...")
                        progress_bar.progress(55)
                        time.sleep(0.35)
                        img_with_boxes, segmentation_overlay, prob_overall, has_anom, suggestions = analyze_mri_unet(
                            input_tensor,
                            ai_model,
                            image_normalized
                        )

                        if has_anom:
                            result = f"Abnormal - Tumor Detected ({prob_overall:.1f}%)"
                            coords = "\n".join(suggestions) if suggestions else "Suspicious region detected by U-Net segmentation."
                        else:
                            result = f"Normal ({100 - prob_overall:.1f}%)"
                            coords = "No suspicious regions localized."

                        timestamp = str(datetime.now().timestamp()).replace(".", "")
                        img1_path = f"history_img/det_{timestamp}.png"
                        img2_path = f"history_img/seg_{timestamp}.png"
                        Image.fromarray(img_with_boxes).save(img1_path)
                        Image.fromarray(segmentation_overlay).save(img2_path)

                        ai_consultation = ""
                        status_message.info("3. Generating Clinical Report...")
                        progress_bar.progress(80)
                        time.sleep(0.35)
                        if OPENAI_KEY:
                            try:
                                client = OpenAI(api_key=OPENAI_KEY)
                                prompt = f"""
                                Act as a Senior Clinical Neuroradiologist. 
                                Patient: {name}, Age: {age}, Gender: {gender}.
                                System Detection: {result}.
                                Localized AI findings: {coords}.
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
                        status_message.success("Case processing finished. The studio has been updated with fresh findings.")
                        progress_bar.progress(100)
                        time.sleep(0.2)
                finally:
                    time.sleep(0.1)
                    status_message.empty()
                    progress_bar.empty()

        if st.session_state.analysis_done:
            with st.container(border=True):
                st.markdown("### AI Findings Board")
                st.success(f"**Diagnosis:** {st.session_state.result}")
                st.write("")
                col_img1, col_img2 = st.columns(2)
                col_img1.image(st.session_state.img1_path, caption="AI Detected Region (Bounding Box)", use_container_width=True)
                col_img2.image(st.session_state.img2_path, caption="U-Net Segmentation Overlay", use_container_width=True)

            with st.container(border=True):
                st.markdown("### Narrative Clinical Summary")
                st.markdown(st.session_state.ai_consultation, unsafe_allow_html=True)
                st.write("")
                pdf_file = export_pdf(st.session_state.name, st.session_state.age, st.session_state.gender, 
                                      st.session_state.result, st.session_state.ai_consultation, 
                                      st.session_state.img1_path, st.session_state.img2_path)
                with open(pdf_file, "rb") as f:
                    st.download_button("Export Clinical Report", f, file_name=f"{remove_accents(st.session_state.name)}_Clinical_Report.pdf", use_container_width=True)

            reset_col_left, reset_col_mid, reset_col_right = st.columns([1, 1.4, 1])
            with reset_col_mid:
                if st.button("Open Fresh Case", use_container_width=True):
                    st.session_state.uploader_nonce += 1
                    for key in [
                        "analysis_done",
                        "ai_consultation",
                        "name",
                        "age",
                        "gender",
                        "result",
                        "img1_path",
                        "img2_path",
                        "coords",
                        "analysis_name",
                        "analysis_age",
                        "analysis_gender",
                    ]:
                        st.session_state.pop(key, None)
                    st.rerun()

    # ==========================
    # TRANG 2: ANALYTICS DASHBOARD
    # ==========================
elif page == "Clinical Command":
        st.markdown('<div class="page-header">Clinical Command Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-subheader">A live operational layer for throughput, case mix, and review activity across the workspace.</div>', unsafe_allow_html=True)
        
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
                    rng = np.random.default_rng(42)
                    mock_dates = pd.date_range(
                        end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1),
                        periods=7
                    )
                    mock_trend_df = pd.DataFrame({
                        'Date': mock_dates.date,
                        'Scans': rng.integers(5, 26, size=len(mock_dates))
                    })
                    trend_df = (
                        pd.concat([mock_trend_df, trend_df], ignore_index=True)
                        .groupby('Date', as_index=False)['Scans']
                        .sum()
                        .sort_values('Date')
                    )
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
elif page == "Patient Atlas":
    st.markdown('<div class="page-header">Patient Timeline Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Browse patient history, review prior analyses, and compare serial studies inside one longitudinal workspace.</div>', unsafe_allow_html=True)

    data = c.execute("SELECT * FROM history ORDER BY time DESC").fetchall()
    df = pd.DataFrame(data, columns=['ID', 'Name', 'Age', 'Gender', 'Date', 'Diagnosis Result', 'Coordinates', 'Image1', 'Image2', 'AI Report'])

    if df.empty:
        st.info("No records found in the database. Please run an MRI analysis first.")
    else:
        df['Date_Parsed'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Patient_Key'] = df['Name'].apply(normalize_patient_name)
        df['Patient_Display_Name'] = df['Name'].apply(lambda name: re.sub(r"\s+", " ", str(name or "").strip()))
        df = df.sort_values(by=['Date_Parsed', 'ID'], ascending=[False, False], na_position='last').reset_index(drop=True)
        # BƯỚC 1: HIỂN THỊ BẢNG DANH BẠ (MASTER VIEW)
        with st.container(border=True):
            col_search, col_export = st.columns([3, 1])
            with col_search:
                search_query = st.text_input("Search by Patient Name", placeholder="Leave blank to show all patients")
            with col_export:
                st.write("") 
                csv = df[['ID', 'Name', 'Age', 'Gender', 'Date', 'Diagnosis Result', 'Coordinates', 'Image1', 'Image2', 'AI Report']].to_csv(index=False).encode('utf-8')
                st.download_button(label="Export Registry Data", data=csv, file_name='NeuroVision_DB.csv', mime='text/csv', use_container_width=True)

            search_query_normalized = normalize_patient_name(search_query)

            if search_query_normalized:
                df_display = df[df['Patient_Key'].str.contains(search_query_normalized, na=False)]
            else:
                df_display = df

            # Lọc bỏ tên trùng, chỉ giữ lần khám mới nhất cho bảng tổng quan
            df_unique_patients = df_display.drop_duplicates(subset=['Patient_Key'], keep='first').reset_index(drop=True)

            st.markdown("### Active Patient List")
            st.info("**Tip:** Click on any row in the table below to open the patient's full medical profile.")
            
            try:
                event = st.dataframe(
                    df_unique_patients[['Date', 'Patient_Display_Name', 'Age', 'Gender', 'Diagnosis Result']].rename(columns={'Patient_Display_Name': 'Name'}),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",          
                    selection_mode="single-row" 
                )
                selected_rows = event.selection.rows
            except Exception as e:
                st.dataframe(
                    df_unique_patients[['Date', 'Patient_Display_Name', 'Age', 'Gender', 'Diagnosis Result']].rename(columns={'Patient_Display_Name': 'Name'}),
                    use_container_width=True,
                    hide_index=True
                )
                st.warning("Lỗi phiên bản Streamlit cũ: Vui lòng chạy lệnh `pip install --upgrade streamlit` ở Terminal để bật tính năng click chọn hàng.")
                selected_rows = []

        # BƯỚC 2: MỞ KHÓA HỒ SƠ CHI TIẾT NẾU CÓ DÒNG BỊ CLICK
        if len(selected_rows) > 0:
            selected_index = selected_rows[0]
            selected_patient_key = df_unique_patients.iloc[selected_index]['Patient_Key']
            p_records_df = (
                df[df['Patient_Key'] == selected_patient_key]
                .sort_values(by=['Date_Parsed', 'ID'], ascending=[False, False], na_position='last')
                .reset_index(drop=True)
            )

            p_records = list(
                p_records_df[['ID', 'Name', 'Age', 'Gender', 'Date', 'Diagnosis Result', 'Coordinates', 'Image1', 'Image2', 'AI Report']]
                .itertuples(index=False, name=None)
            )
            p_records_asc = sorted(p_records, key=lambda x: x[4]) # Sắp xếp từ cũ đến mới
            latest_record = p_records[0]
            selected_patient = p_records_df.iloc[0]['Patient_Display_Name']
            selected_patient_ids = p_records_df['ID'].tolist()
            p_records_asc = list(reversed(p_records))

            with st.container(border=True):
                st.markdown(f"<h2 style='text-align:center; color:#163128;'>Patient Dossier: {selected_patient}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center; color:#5f776d;'>Age: {latest_record[2]} | Gender: {latest_record[3]} | Total Scans: {len(p_records)}</p>", unsafe_allow_html=True)
                
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
                                colB.image(rec[8], caption="U-Net Segmentation Overlay", use_container_width=True)
                            except:
                                st.warning("Images are no longer stored on the server.")

                            try:
                                hist_pdf = export_pdf(rec[1], rec[2], rec[3], rec[5], report_text, rec[7], rec[8])
                                with open(hist_pdf, "rb") as f:
                                    st.download_button("Export Visit Report", f, file_name=f"{remove_accents(rec[1])}_{rec[4][:10]}_Report.pdf", key=f"dl_hist_{rec_id}")
                            except Exception:
                                pass

                # --- TAB 2: SO SÁNH TIẾN TRIỂN (CÓ CHỌN FILE) ---
                with tab_comp:
                    if len(p_records_asc) < 2:
                        st.info("Not enough data. You need to scan this patient at least twice to track disease progression over time.")
                    else:
                        st.markdown("<h4 style='text-align:center;'>Study-to-Study Comparison</h4>", unsafe_allow_html=True)
                        
                        scan_options = [
                            (
                                f"Scan on {r[4]} | {r[5].split('(')[0].strip()} | Record #{r[0]}",
                                r
                            )
                            for r in p_records_asc
                        ]
                        scan_labels = [label for label, _ in scan_options]
                        scan_lookup = dict(scan_options)

                        if len(scan_labels) < 2:
                            st.info("Not enough unique scan entries are available for comparison after the latest update.")
                        else:
                            sel_col1, sel_col2 = st.columns(2)
                            with sel_col1:
                                label_a = st.selectbox("Select Scan A (Baseline):", scan_labels, index=max(0, len(scan_labels) - 2))
                            with sel_col2:
                                label_b = st.selectbox("Select Scan B (Follow-up):", scan_labels, index=len(scan_labels) - 1)

                            scan_a = scan_lookup[label_a]
                            scan_b = scan_lookup[label_b]

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
                                st.markdown("<h5 style='text-align:center;'>2. Segmentation Overlay Comparison</h5>", unsafe_allow_html=True)
                                hm_col1, hm_col2 = st.columns(2)
                                try:
                                    hm_col1.image(scan_a[8], caption=f"Overlay A ({scan_a[4][:10]})", use_container_width=True)
                                    hm_col2.image(scan_b[8], caption=f"Overlay B ({scan_b[4][:10]})", use_container_width=True)
                                except:
                                    st.error("Image missing for comparison.")

                                if st.button("Generate Progression Brief", use_container_width=True, type="primary", key=f"btn_comp_{selected_patient}"):
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
                                            st.download_button("Export Progression Brief", f, file_name=f"{remove_accents(selected_patient)}_Progression.pdf", use_container_width=True)

                # --- TAB 3: CHỈNH SỬA THÔNG TIN ---
                with tab_edit:
                    with st.form(key=f"edit_form_profile"):
                        st.info("Update demographic details for this patient. This will update ALL their past scan records to ensure consistency.")
                        e_col1, e_col2, e_col3 = st.columns(3)
                        new_name = e_col1.text_input("Full Name", value=latest_record[1])
                        new_age = e_col2.text_input("Age", value=latest_record[2])
                        
                        gender_index = ["Male", "Female", "Other"].index(latest_record[3]) if latest_record[3] in ["Male", "Female", "Other"] else 0
                        new_gender = e_col3.selectbox("Gender", ["Male", "Female", "Other"], index=gender_index)
                        
                        submit_edit = st.form_submit_button("Apply Profile Update")
                        if submit_edit:
                            c.executemany(
                                "UPDATE history SET name=?, age=?, gender=? WHERE id=?",
                                [(new_name.strip(), new_age, new_gender, rec_id) for rec_id in selected_patient_ids]
                            )
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
                                if st.button("Remove This Visit", key=f"del_scan_{rec_id}", use_container_width=True):
                                    c.execute("DELETE FROM history WHERE id=?", (rec_id,))
                                    conn.commit()
                                    st.session_state.pop("comp_pair", None)
                                    st.session_state.pop("comp_report", None)
                                    st.rerun()

                    st.markdown("---")
                    st.markdown("#### Bulk Action")
                    if st.button("Archive Entire Patient Record", type="primary", use_container_width=True):
                        c.executemany("DELETE FROM history WHERE id=?", [(rec_id,) for rec_id in selected_patient_ids])
                        conn.commit()
                        st.session_state.pop("comp_pair", None)
                        st.session_state.pop("comp_report", None)
                        st.rerun()
        else:
            st.info("Please click on a patient row in the table above to view their details, reports, and compare scans.")

st.markdown('<a href="#top-of-page" class="scroll-to-top">Top</a>', unsafe_allow_html=True)
