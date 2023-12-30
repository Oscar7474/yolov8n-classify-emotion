from PyQt5.QtGui import QPixmap
from ultralytics import YOLO
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from UI import Ui_Form
import sys

Fer_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


class MyWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

    def check(self):
        img = img_path + "\\" + self.lineEdit.text() + ".jpg"
        self.label_3.setPixmap(QPixmap(img).scaled(96, 96))
        results = model.predict(img)
        self.label_2.setText(Fer_dict[results[0].probs.top1])


if __name__ == "__main__":
    model = YOLO(r"C:\Users\Kuo\Downloads\CLS\CLS\last.pt")
    img_path = r"C:\Users\Kuo\Downloads\CLS\CLS\Fer2013\list"
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWindow()
    Form.setWindowTitle("Fer_2013")
    Form.show()
    sys.exit(app.exec_())
