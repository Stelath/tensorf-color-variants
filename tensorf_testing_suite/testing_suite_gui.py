# import torch

from PyQt6.QtWidgets import QApplication, QLabel, QWidget
import sys

import tensorf_testing_suite

def main():
    app = QApplication([])
    window = QWidget()
    window.setWindowTitle("Tensorf Testing Suite")
    window.setGeometry(100, 100, 280, 80)
    helloMsg = QLabel("<h1>Tensorf Testing Suite!</h1>", parent=window)
    helloMsg.move(60, 15)
    window.show()
    sys.exit(app.exec())
