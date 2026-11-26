from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QLineEdit,
    QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import wtr_data
import wtr_model

class GUI(QMainWindow):
    def __init__(self):
        self.curr_coll = ""

        super().__init__()
        self.setWindowTitle("wheretoread")
        self.resize(1024, 768)

        self.chroma_cl = wtr_data.cl

        self.main_layout = QVBoxLayout()
        window = QWidget()
        window.setLayout(self.main_layout)

        self.main_layout.addWidget(QLabel("wheretoread", font=QFont("Segoe UI", 16)), alignment=Qt.AlignmentFlag.AlignHCenter)
        self.curr_coll_label = QLabel("Current collection: NULL")
        self.main_layout.addWidget(self.curr_coll_label)

        self.load_coll_widget = QWidget()
        self.load_coll_layout = QVBoxLayout()
        self.load_coll_widget.setLayout(self.load_coll_layout)

        file_upload = QPushButton("Upload document")
        file_upload.clicked.connect(self.open_file_dialog)
        self.load_coll_layout.addWidget(file_upload)

        self.collections_list = QListWidget()
        self.collections_list.addItems([c.name for c in self.chroma_cl.list_collections()])
        self.collections_list.setMaximumHeight(100)
        self.load_coll_layout.addWidget(self.collections_list)

        load_collection_button = QPushButton("Use this collection")
        load_collection_button.clicked.connect(self.load_collection)
        self.load_coll_layout.addWidget(load_collection_button)

        self.load_coll_layout.addStretch()
        self.main_layout.addWidget(self.load_coll_widget)

        self.chat_widget = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_widget.setLayout(self.chat_layout)

        self.chat_layout.addWidget(QLabel("Ask", font=QFont("Segoe UI", 13)))

        self.chat_input = QLineEdit()
        self.chat_input.returnPressed.connect(self.ui_answer)
        self.chat_layout.addWidget(self.chat_input)

        self.chat_submit_btn = QPushButton("Query")
        self.chat_submit_btn.clicked.connect(self.ui_answer)
        self.chat_layout.addWidget(self.chat_submit_btn)

        self.chat_answer = QLabel(wordWrap=True)
        self.chat_layout.addWidget(self.chat_answer)

        self.chat_layout.addStretch()
        self.chat_widget.setDisabled(True)
        self.main_layout.addWidget(self.chat_widget)

        self.main_layout.addStretch()
        self.setCentralWidget(window)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select document", "",
            "PDF (*.pdf)"
        )

        if file_path:
            logger.info(f"Got file: {file_path}")
            wtr_data.build(file_path)

            self.collections_list.clear()
            self.collections_list.addItems([c.name for c in self.chroma_cl.list_collections()])

    def load_collection(self):
        self.curr_coll = self.collections_list.currentItem().text()
        self.main_layout.removeWidget(self.load_coll_widget)
        self.load_coll_widget.deleteLater()

        self.curr_coll_label.setText(f"Current collection: {self.curr_coll}")

        self.chat_widget.setEnabled(True)

    def ui_answer(self):
        q = self.chat_input.text()
        res = wtr_model.answer(q, 2, self.curr_coll)

        self.chat_answer.setText(res)

def run_app():
    try:
        ollama_proc = wtr_model.begin_ollama()

        app = QApplication([])
        window = GUI()
        window.show()
        app.exec()
    except Exception as e:
        logger.error(f"Error in starting app: {e}")
    finally:
        ollama_proc.terminate()