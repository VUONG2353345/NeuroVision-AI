import streamlit as st
import sqlite3
from datetime import datetime
from PIL import Image
import numpy as np
import pydicom
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import os
import torch
import cv2
from torchvision import transforms
import sys

# Trỏ đường dẫn vào thư mục chứa code AI của bạn
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mri_analyzer import analyze_brain_ai_driven
from mri_model import ResNetMRI
# =========================
# CONFIG & FOLDERS
# =========================
st.set_page_config(page_title="MediAI", layout="wide", page_icon="🧠")

# Tạo folder lưu ảnh lịch sử nếu chưa có
if not os.path.exists("history_img"):
    os.makedirs("history_img")

# =========================
# DB SETUP (ĐÃ NÂNG CẤP)
# =========================
@st.cache_resource
def load_ai_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetMRI(num_classes=2).to(device)
    # Đảm bảo đường dẫn này đúng với vị trí file .pth của bạn
    model.load_state_dict(torch.load("models/mri_model_best.pth", map_location=device))
    model.eval()
    return model, device

# Khởi tạo model sẵn khi web vừa bật lên
ai_model, device = load_ai_model()
conn = sqlite3.connect("mediai.db", check_same_thread=False)
c = conn.cursor()

# Thêm cột coords, img1_path, img2_path
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
conn.commit()

# =========================
# CSS NÂNG CAO (HIỆU ỨNG MƯỢT MÀ)
# =========================
st.markdown("""
<style>
.big-title {
    font-size: 40px;
    font-weight: 800;
    background: -webkit-linear-gradient(45deg, #00ADB5, #0056b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
}
/* Can thiệp khung tải file */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #00ADB5 !important;
    border-radius: 15px !important;
    padding: 40px !important;
    background-color: rgba(0, 173, 181, 0.05);
    transition: all 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: rgba(0, 173, 181, 0.15);
    transform: scale(1.01);
}
/* Hiệu ứng mượt mà cho ảnh */
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(15px); }
    100% { opacity: 1; transform: translateY(0); }
}
.stImage > img {
    animation: fadeIn 0.6s ease-out;
    border-radius: 12px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.15);
    transition: transform 0.3s ease;
}
.stImage > img:hover {
    transform: scale(1.02);
}
/* Card thông tin */
.info-card {
    background-color: #2b2b36;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #00ADB5;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧠 MediAI System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🔬 Analysis Workspace", "📊 Patient History"])

# =========================
# FAKE MODEL & PDF
# =========================
def analyze_mri(dicom):
    pass # Chuyển logic xuống dưới để dễ lưu file

def export_pdf(name, result, img1_path, img2_path):
    file = f"report_{datetime.now().timestamp()}.pdf"
    doc = SimpleDocTemplate(file)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph(f"Patient: {name}", styles["Heading1"]))
    elements.append(Paragraph(f"Result: {result}", styles["Heading2"]))
    elements.append(RLImage(img1_path, width=200, height=200))
    elements.append(RLImage(img2_path, width=200, height=200))
    doc.build(elements)
    return file

# =========================
# ANALYSIS PAGE
# =========================
if page == "🔬 Analysis Workspace":
    st.markdown('<div class="big-title">MRI Diagnostic Workspace</div>', unsafe_allow_html=True)

    st.subheader("👤 Patient Information")
    col1, col2, col3 = st.columns(3)
    name = col1.text_input("Full Name", placeholder="e.g. Nguyen Viet Tan Vuong")
    age = col2.text_input("Age", placeholder="e.g. 21")
    gender = col3.selectbox("Gender", ["Male", "Female", "Other"])

    st.markdown("---")
    st.subheader("📤 Upload Scan")
    file = st.file_uploader("Supported formats: .dcm, .png, .jpg", type=["dcm", "ima", "png", "jpg", "jpeg"])

    with st.spinner("🤖 Processing deep learning model..."):
                # Đọc ảnh gốc bằng OpenCV để giống với luồng train model
                file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                image = image.astype(np.float32)

                # Tiền xử lý
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                input_tensor = transform(image_resized).unsqueeze(0).to(device)

                # 👉 CHẠY AI THẬT
                img_with_boxes, heatmap, prob_overall, has_anom, suggestions = analyze_brain_ai_driven(input_tensor, ai_model, image_resized)

                # Trích xuất kết quả
                if has_anom:
                    result = f"Abnormal - Tumor Detected ({prob_overall:.1f}%)"
                    coords = "Detected by Grad-CAM" # Có thể thay bằng tọa độ thật nếu hàm AI có trả về bounding box
                else:
                    result = f"Normal ({100 - prob_overall:.1f}%)"
                    coords = "None"

                # Generate unique filenames for history
                timestamp = str(datetime.now().timestamp()).replace(".", "")
                img1_path = f"history_img/det_{timestamp}.png"
                img2_path = f"history_img/heat_{timestamp}.png"
                
                # Lưu ảnh đầu ra từ AI (img_with_boxes và heatmap thường là numpy array màu RGB)
                Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)).save(img1_path)
                Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)).save(img2_path)

# =========================
# HISTORY PAGE (ĐÃ LÀM LẠI HOÀN TOÀN)
# =========================
elif page == "📊 Patient History":
    st.markdown('<div class="big-title">Medical Records Database</div>', unsafe_allow_html=True)

    data = c.execute("SELECT * FROM history ORDER BY id DESC").fetchall()

    if not data:
        st.info("No records found. Please run an analysis first.")
    else:
        # Dashboard Overview
        total = len(data)
        abnormal = len([d for d in data if "Abnormal" in d[5]])
        normal = total - abnormal

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scans", total)
        col2.metric("Critical Cases (Abnormal)", abnormal, delta="-Urgent", delta_color="inverse")
        col3.metric("Normal Cases", normal, delta="+Safe", delta_color="normal")
        st.markdown("---")

        # Patient List with Images
        for d in data:
            # d index: 0:id, 1:name, 2:age, 3:gender, 4:time, 5:result, 6:coords, 7:img1, 8:img2
            status_icon = "🚨" if "Abnormal" in d[5] else "✅"
            
            with st.expander(f"{status_icon} {d[4]} | Patient: {d[1]} (Age: {d[2]}) - {d[5]}"):
                
                # Khung thông tin tọa độ
                st.markdown(f"""
                <div class="info-card">
                    <b>Gender:</b> {d[3]} <br>
                    <b>AI Coordinates:</b> <code>{d[6]}</code>
                </div>
                """, unsafe_allow_html=True)
                
                # Load ảnh từ folder
                colA, colB = st.columns(2)
                try:
                    colA.image(d[7], caption="Detected Region")
                    colB.image(d[8], caption="Attention Heatmap")
                except:
                    st.error("Image file not found on server.")