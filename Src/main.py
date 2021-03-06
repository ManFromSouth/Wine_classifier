import sys

from PySide2.QtWidgets import QApplication, QMainWindow

from main_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

sys.exit(app.exec_())
