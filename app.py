import sys
import os
import torch
import numpy as np
import cv2  
import tempfile # [TÍNH NĂNG MỚI] Tạo file tạm cho PDF
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QPixmap, QTextDocument # [TÍNH NĂNG MỚI] QTextDocument để tạo PDF
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer, QMarginsF  # Bổ sung QMarginsF
from PyQt6.QtGui import QPixmap, QTextDocument, QPageLayout          # Bổ sung QPageLayout
from PyQt6.QtPrintSupport import QPrinter # [TÍNH NĂNG MỚI] Hỗ trợ in PDF
from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QVBoxLayout,
                             QHBoxLayout, QWidget, QGridLayout, QProgressBar,
                             QLabel, QPushButton, QStackedWidget, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pydicom

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from mri_analyzer import analyze_brain_ai_driven
from mri_model import ResNetMRI

# ====================== Canvas hiển thị MRI ======================
class MRICanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(10, 5), facecolor='white')
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_heat = self.fig.add_subplot(122)
        self.fig.tight_layout(pad=2.0)
        super().__init__(self.fig)

    def plot_image(self, img_with_boxes, heatmap, title=""):
        self.ax_orig.clear()
        self.ax_heat.clear()
        
        self.ax_orig.imshow(img_with_boxes)
        self.ax_orig.set_title(f"MRI & Suspected Regions\n[File: {title}]", fontsize=11, fontweight='bold', color='#2c3e50')
        self.ax_orig.tick_params(axis='both', labelsize=8)
        self.ax_orig.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        self.ax_heat.imshow(heatmap)
        self.ax_heat.set_title(f"AI Activation Map (Grad-CAM)\n[File: {title}]", fontsize=11, fontweight='bold', color='#2c3e50')
        self.ax_heat.tick_params(axis='both', labelsize=8)
        self.ax_heat.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        self.draw()

# ====================== Worker xử lý MRI ======================
class MRIPredictWorker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(str, np.ndarray, np.ndarray, str) 

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load mô hình Supervised AI
            model = ResNetMRI(num_classes=2).to(device)
            model.load_state_dict(torch.load("models/mri_model_best.pth", map_location=device))
            model.eval()

            # Đọc file
            ext = self.file_path.lower()
            if ext.endswith(('.ima', '.dcm')):
                ds = pydicom.dcmread(self.file_path)
                image = ds.pixel_array.astype(np.float32)
            else:
                image = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
                if image is None: raise Exception("Không hỗ trợ định dạng ảnh này.")
                image = image.astype(np.float32)

            if image.ndim == 3:
                image = image[image.shape[0] // 2, :, :]
            
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            input_tensor = transform(image_resized).unsqueeze(0).to(device)

            for i in range(1, 101):
                self.progress.emit(i)
                QThread.msleep(5)

            # Phân tích
            img_with_boxes, heatmap, prob_overall, has_anom, suggestions = analyze_brain_ai_driven(input_tensor, model, image_resized)

            if has_anom:
                details = "\n".join(suggestions)
                result_text = (f"⚠️ AI DETECTED ABNORMALITY (Confidence: {prob_overall:.1f}%)\n"
                               f"----------------------------------------------------\n"
                               f"{details}\n"
                               f"----------------------------------------------------\n"
                               f"👉 Heatmap generated directly from AI's deep neural layers.")
            else:
                result_text = (f"✅ NO ABNORMALITY DETECTED\n"
                               f"AI Confidence for normal structure: {(100 - prob_overall):.1f}%\n"
                               f"The deep learning model found no pathological features.")

            self.done.emit(result_text, img_with_boxes, heatmap, os.path.basename(self.file_path))
        except Exception as e:
            print(f"Error: {e}")
            self.done.emit(f"❌ Processing Error: {str(e)}", np.zeros((224,224,3), dtype=np.uint8), np.zeros((224,224,3), dtype=np.uint8), "Error")

# ====================== Cửa sổ chính ======================
class SeizureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hệ thống Kiểm tra Động kinh & MRI - HCMUT")
        self.resize(1100, 800)
        self.setStyleSheet("background-color: white; color: black;")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        
        # [TÍNH NĂNG MỚI] Biến lưu trữ dữ liệu để xuất file
        self.current_file_name = ""
        self.current_result_text = ""

        # ---------- Trang 1: Màn hình chính ----------
        self.home_page = QWidget()
        home_layout = QVBoxLayout(self.home_page)
        home_layout.setContentsMargins(20, 20, 20, 20)

        brain_logo = QLabel()
        pixmap = QPixmap("brain.png")
        if not pixmap.isNull():
            pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            brain_logo.setPixmap(pixmap)
        else:
            brain_logo.setText("🧠")
            brain_logo.setStyleSheet("font-size: 80px; color: #2c3e50;")
        brain_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        home_layout.addWidget(brain_logo)

        title = QLabel("HỆ THỐNG CHẨN ĐOÁN HÌNH ẢNH NÃO")
        title.setStyleSheet("font-size: 36px; font-weight: bold; color: #2c3e50; margin: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        home_layout.addWidget(title)

        self.btn_mri = QPushButton("📁 NẠP FILE MRI MỌI ĐỊNH DẠNG")
        self.btn_mri.setFixedSize(340, 70)
        self.btn_mri.setStyleSheet("QPushButton { border: 3px solid #e67e22; border-radius: 20px; font-size: 18px; font-weight: bold; background-color: white; color: #2c3e50; } QPushButton:hover { background-color: #f39c12; color: white; }")
        self.btn_mri.clicked.connect(self.load_mri)
        home_layout.addWidget(self.btn_mri, alignment=Qt.AlignmentFlag.AlignCenter)

        home_layout.addStretch()

        footer = QHBoxLayout()
        dev_label = QLabel("Developed by HCMUTer")
        dev_label.setStyleSheet("font-size: 16px; color: #7f8c8d;")
        footer.addWidget(dev_label)
        footer.addStretch()
        bku_logo = QLabel()
        bku_pix = QPixmap("bku_logo.png")
        if not bku_pix.isNull():
            bku_logo.setPixmap(bku_pix.scaled(50, 50, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            bku_logo.setText("BKU")
        bku_logo.setAlignment(Qt.AlignmentFlag.AlignRight)
        footer.addWidget(bku_logo)
        home_layout.addLayout(footer)
        self.stack.addWidget(self.home_page)

        # Style cho nút bấm
        btn_style = "QPushButton { border: 2px solid #7f8c8d; border-radius: 10px; padding: 12px 30px; font-size: 16px; font-weight: bold; background-color: white; color: black; } QPushButton:hover { background-color: #ecf0f1; }"

        # ---------- Trang 2: MRI ----------
        self.mri_page = QWidget()
        mri_layout = QGridLayout(self.mri_page)
        self.mri_canvas = MRICanvas(self)
        mri_layout.addWidget(self.mri_canvas, 0, 0, 1, 3)

        self.mri_progress = QProgressBar()
        self.mri_progress.setStyleSheet("QProgressBar { border: 2px solid #bdc3c7; border-radius: 8px; text-align: center; height: 25px; background-color: white; } QProgressBar::chunk { background-color: #e67e22; border-radius: 8px; }")
        self.mri_progress.setVisible(False)
        mri_layout.addWidget(self.mri_progress, 1, 0, 1, 3)

        self.mri_result = QLabel("")
        self.mri_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mri_layout.addWidget(self.mri_result, 2, 0, 1, 3)

        nav_mri = QHBoxLayout()
        
        # [TÍNH NĂNG MỚI] Giao diện nút Xuất Báo Cáo
        self.btn_export_pdf = QPushButton("📄 Xuất Báo Cáo PDF")
        self.btn_export_img = QPushButton("🖼 Xuất Ảnh Kết Quả")
        self.btn_export_pdf.setStyleSheet(btn_style + "QPushButton { color: #2980b9; border-color: #2980b9; } QPushButton:hover { background-color: #2980b9; color: white; }")
        self.btn_export_img.setStyleSheet(btn_style + "QPushButton { color: #8e44ad; border-color: #8e44ad; } QPushButton:hover { background-color: #8e44ad; color: white; }")
        
        # Mặc định vô hiệu hóa cho đến khi AI chạy xong
        self.btn_export_pdf.setEnabled(False)
        self.btn_export_img.setEnabled(False)
        
        self.btn_export_pdf.clicked.connect(self.export_pdf_report)
        self.btn_export_img.clicked.connect(self.export_image)

        btn_back_mri = QPushButton("⬅ Quay lại")
        btn_exit_mri = QPushButton("✖ Thoát")
        btn_back_mri.setStyleSheet(btn_style)
        btn_exit_mri.setStyleSheet(btn_style + "QPushButton:hover { background-color: #e74c3c; color: white; }")
        btn_back_mri.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_exit_mri.clicked.connect(self.close)
        
        nav_mri.addWidget(self.btn_export_pdf)
        nav_mri.addWidget(self.btn_export_img)
        nav_mri.addStretch()
        nav_mri.addWidget(btn_back_mri)
        nav_mri.addWidget(btn_exit_mri)
        
        mri_layout.addLayout(nav_mri, 3, 0, 1, 3)
        mri_layout.setRowStretch(0, 10)
        mri_layout.setRowStretch(2, 1)
        
        self.stack.addWidget(self.mri_page)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide_progress)

    def load_mri(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file MRI", "", "Image Files (*.ima *.dcm *.jpg *.jpeg *.png)")
        if file_path:
            self.stack.setCurrentIndex(1)
            self.mri_progress.setValue(0)
            self.mri_progress.setVisible(True)
            self.mri_result.setText("🔍 Scanning brain structure...")
            self.mri_result.setStyleSheet("font-size: 28px; font-weight: bold; color: #e67e22;")
            
            # [TÍNH NĂNG MỚI] Vô hiệu hóa nút xuất khi đang chạy ảnh mới
            self.btn_export_pdf.setEnabled(False)
            self.btn_export_img.setEnabled(False)

            self.mri_worker = MRIPredictWorker(file_path)
            self.mri_worker.progress.connect(self.mri_progress.setValue)
            self.mri_worker.done.connect(self.show_mri_result)
            self.mri_worker.start()

    def show_mri_result(self, result_text, img_with_boxes, heatmap, file_name):
        if img_with_boxes is not None:
            self.mri_canvas.plot_image(img_with_boxes, heatmap, title=file_name)
        else:
            self.mri_canvas.ax_orig.clear()
            self.mri_canvas.ax_heat.clear()
            self.mri_canvas.ax_orig.text(0.5, 0.5, "Lỗi hiển thị", ha='center', va='center')
            self.mri_canvas.draw()

        self.mri_result.setText(result_text)
        color = "#e74c3c" if "ABNORMALITY" in result_text or "BẤT THƯỜNG" in result_text else "#27ae60"
        self.mri_result.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color}; margin: 10px;")
        
        # [TÍNH NĂNG MỚI] Lưu dữ liệu và bật nút xuất
        self.current_file_name = file_name
        self.current_result_text = result_text
        self.btn_export_pdf.setEnabled(True)
        self.btn_export_img.setEnabled(True)
        
        self.timer.start(2000)

    def hide_progress(self):
        self.mri_progress.setVisible(False)

    # =========================================================================
    # [TÍNH NĂNG MỚI] CÁC HÀM XUẤT FILE
    # =========================================================================
    def export_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu Ảnh Kết Quả", f"Ket_Qua_AI_{self.current_file_name}.png", "PNG Files (*.png);;JPEG Files (*.jpg)")
        if file_path:
            # Lưu trực tiếp Matplotlib Figure (chứa cả 2 ảnh cạnh nhau)
            self.mri_canvas.fig.savefig(file_path, bbox_inches='tight', dpi=300)
            QMessageBox.information(self, "Thành công", f"Đã lưu ảnh phân tích tại:\n{file_path}")

    def export_pdf_report(self):
        # BÍ QUYẾT: Cắt bỏ đuôi .jpg/.png của file gốc để tên PDF trông sạch sẽ
        import os
        base_name = os.path.splitext(self.current_file_name)[0]
        default_pdf_name = f"Bao_Cao_Y_Te_{base_name}.pdf"
        
        file_path, _ = QFileDialog.getSaveFileName(self, "Lưu Báo Cáo Y Tế PDF", default_pdf_name, "PDF Files (*.pdf)")
        if file_path:
            # 1. Lưu tạm ảnh Canvas để nhúng vào PDF
            import tempfile
            temp_img = tempfile.mktemp(suffix=".png")
            self.mri_canvas.fig.savefig(temp_img, bbox_inches='tight', pad_inches=0.1, dpi=200)
            img_path = temp_img.replace('\\', '/')

            # Chuyển đổi ký tự xuống dòng của Python thành thẻ <br> của HTML
            formatted_result = self.current_result_text.replace('\n', '<br>')

            # 2. Định dạng HTML (Thu nhỏ ảnh xuống width=600 để nằm gọn trên 1 trang)
            html_content = f"""
            <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h1 style="text-align: center; color: #2c3e50; font-size: 22pt;">BÁO CÁO CHẨN ĐOÁN HÌNH ẢNH (MRI AI)</h1>
                <hr style="border: 1px solid #bdc3c7;">
                
                <h3 style="font-size: 14pt;">THÔNG TIN PHIÊN QUÉT</h3>
                <ul style="font-size: 12pt; line-height: 1.5;">
                    <li><b>Tệp nguồn:</b> {self.current_file_name}</li>
                    <li><b>Ghi chú:</b> Quét bằng mạng Nơ-ron Tích chập (CNN - ResNet) kết hợp Grad-CAM</li>
                </ul>

                <h3 style="color: #c0392b; margin-top: 20px; font-size: 14pt;">KẾT QUẢ PHÂN TÍCH TỪ AI</h3>
                <table width="100%" style="background-color: #fef9f9; border-left: 5px solid #e74c3c;">
                    <tr>
                        <td style="padding: 15px;">
                            <p style="font-size: 14pt; font-family: 'Courier New', monospace; font-weight: bold; color: #c0392b; margin: 0;">
                                {formatted_result}
                            </p>
                        </td>
                    </tr>
                </table>

                <h3 style="margin-top: 30px; font-size: 14pt;">HÌNH ẢNH VÀ BẢN ĐỒ NHIỆT LÂM SÀNG</h3>
                <table width="100%">
                    <tr>
                        <td align="center">
                            <img src="{img_path}" width="600">
                        </td>
                    </tr>
                </table>
                
                <br>
                <hr style="border: 1px solid #bdc3c7;">
                <p style="text-align: right; font-size: 10pt; color: #7f8c8d;">
                    <i>*Báo cáo phát sinh tự động bởi Hệ thống AI HCMUT.<br>Vui lòng đối chiếu lâm sàng trước khi kết luận.</i>
                </p>
            </body>
            </html>
            """

            # 3. Sử dụng QTextDocument để chuyển HTML thành PDF
            doc = QTextDocument()
            doc.setHtml(html_content)

            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(file_path)
            
            # Cài đặt lề giấy A4
            printer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)
            
            # Khởi chạy in
            doc.print(printer)

            # Xóa file ảnh tạm
            if os.path.exists(temp_img):
                os.path.exists(temp_img) and os.remove(temp_img)
                
            QMessageBox.information(self, "Thành công", f"Báo cáo PDF y tế đã được tạo thành công tại:\n{file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SeizureApp()
    window.show()
    sys.exit(app.exec())