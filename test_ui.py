import sys
from PyQt6 import QtWidgets, uic

class TestUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Nạp đúng cái file giao diện bạn vừa thiết kế nãy giờ
        uic.loadUi("app_desktop.ui", self) 
        
        # Kết nối 2 nút bấm với tính năng lật trang
        self.btn_analysis.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.btn_history.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TestUI()
    window.show()
    sys.exit(app.exec())