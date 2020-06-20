# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_ui.ui',
# licensing of 'main_ui.ui' applies.
#
# Created: Fri Jun 12 22:25:28 2020
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!

import numpy as np
from PySide2 import QtCore, QtGui, QtWidgets

from Models import Models


# заменяет все запятые в строке точками
def convert_colons(string):
    return string.replace(',', '.')

class Ui_MainWindow(object):
    def __init__(self):
        # super(self).__init__()
        # инициализация классификаторов
        self.models = Models()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1038, 593)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(419, 240, 561, 261))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        # зона результатов
        self.label_nbc_type = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_nbc_type.setText("")
        self.label_nbc_type.setObjectName("label_nbc_type")
        self.label_nbc_type.setAlignment(QtCore.Qt.AlignCenter)
        self.label_nbc_type.setVisible(False)
        self.gridLayout.addWidget(self.label_nbc_type, 3, 1, 1, 1)
        self.label_svm_zone = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_svm_zone.setText("")
        self.label_svm_zone.setObjectName("label_svm_zone")
        self.label_svm_zone.setAlignment(QtCore.Qt.AlignCenter)
        self.label_svm_zone.setVisible(False)
        self.gridLayout.addWidget(self.label_svm_zone, 1, 2, 1, 1)
        self.label_knn_zone = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_knn_zone.setText("")
        self.label_knn_zone.setObjectName("label_knn_zone")
        self.label_knn_zone.setAlignment(QtCore.Qt.AlignCenter)
        self.label_knn_zone.setVisible(False)
        self.gridLayout.addWidget(self.label_knn_zone, 2, 2, 1, 1)
        self.label_nbc_zone = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_nbc_zone.setText("")
        self.label_nbc_zone.setObjectName("label_nbc_zone")
        self.label_nbc_zone.setAlignment(QtCore.Qt.AlignCenter)
        self.label_nbc_zone.setVisible(False)
        self.gridLayout.addWidget(self.label_nbc_zone, 3, 2, 1, 1)
        self.label_svm_head = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_svm_head.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_svm_head.setObjectName("label_svm_head")
        self.label_svm_head.setVisible(False)
        self.gridLayout.addWidget(self.label_svm_head, 1, 0, 1, 1)
        self.label_type_head = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_type_head.setObjectName("label_type_head")
        self.label_type_head.setAlignment(QtCore.Qt.AlignCenter)
        self.label_type_head.setVisible(False)
        self.gridLayout.addWidget(self.label_type_head, 0, 1, 1, 1)
        self.label_knn_head = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_knn_head.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_knn_head.setObjectName("label_knn_head")
        self.label_knn_head.setVisible(False)
        self.gridLayout.addWidget(self.label_knn_head, 2, 0, 1, 1)
        self.label_zone_head = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_zone_head.setObjectName("label_zone_head")
        self.label_zone_head.setAlignment(QtCore.Qt.AlignCenter)
        self.label_zone_head.setVisible(False)
        self.gridLayout.addWidget(self.label_zone_head, 0, 2, 1, 1)
        self.label_nbc_head = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_nbc_head.setAlignment(QtCore.Qt.AlignCenter)
        self.label_nbc_head.setObjectName("label_nbc_head")
        self.label_nbc_head.setVisible(False)
        self.gridLayout.addWidget(self.label_nbc_head, 3, 0, 1, 1)
        self.label_knn_type = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_knn_type.setText("")
        self.label_knn_type.setObjectName("label_knn_type")
        self.label_knn_type.setAlignment(QtCore.Qt.AlignCenter)
        self.label_knn_type.setVisible(False)
        self.gridLayout.addWidget(self.label_knn_type, 2, 1, 1, 1)
        self.label_svm_type = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_svm_type.setText("")
        self.label_svm_type.setObjectName("label_svm_type")
        self.label_svm_type.setAlignment(QtCore.Qt.AlignCenter)
        self.label_svm_type.setVisible(False)
        # зона вводов
        self.gridLayout.addWidget(self.label_svm_type, 1, 1, 1, 1)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(120, 110, 251, 451))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_Al = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Al.setFont(font)
        self.label_Al.setTabletTracking(True)
        self.label_Al.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Al.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Al.setObjectName("label_Al")
        self.verticalLayout.addWidget(self.label_Al)
        self.label_Ba = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Ba.setFont(font)
        self.label_Ba.setTabletTracking(True)
        self.label_Ba.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Ba.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Ba.setObjectName("label_Ba")
        self.verticalLayout.addWidget(self.label_Ba)
        self.label_Ca = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Ca.setFont(font)
        self.label_Ca.setTabletTracking(True)
        self.label_Ca.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Ca.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Ca.setObjectName("label_Ca")
        self.verticalLayout.addWidget(self.label_Ca)
        self.label_Cu = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Cu.setFont(font)
        self.label_Cu.setTabletTracking(True)
        self.label_Cu.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Cu.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Cu.setObjectName("label_Cu")
        self.verticalLayout.addWidget(self.label_Cu)
        self.label_Fe = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Fe.setFont(font)
        self.label_Fe.setTabletTracking(True)
        self.label_Fe.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Fe.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Fe.setObjectName("label_Fe")
        self.verticalLayout.addWidget(self.label_Fe)
        self.label_K = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_K.setFont(font)
        self.label_K.setTabletTracking(True)
        self.label_K.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_K.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_K.setObjectName("label_K")
        self.verticalLayout.addWidget(self.label_K)
        self.label_Li = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Li.setFont(font)
        self.label_Li.setTabletTracking(True)
        self.label_Li.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Li.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Li.setObjectName("label_Li")
        self.verticalLayout.addWidget(self.label_Li)
        self.label_Mg = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Mg.setFont(font)
        self.label_Mg.setTabletTracking(True)
        self.label_Mg.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Mg.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Mg.setObjectName("label_Mg")
        self.verticalLayout.addWidget(self.label_Mg)
        self.label_Mn = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Mn.setFont(font)
        self.label_Mn.setTabletTracking(True)
        self.label_Mn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Mn.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Mn.setObjectName("label_Mn")
        self.verticalLayout.addWidget(self.label_Mn)
        self.label_Na = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Na.setFont(font)
        self.label_Na.setTabletTracking(True)
        self.label_Na.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Na.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Na.setObjectName("label_Na")
        self.verticalLayout.addWidget(self.label_Na)
        self.label_Ni = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Ni.setFont(font)
        self.label_Ni.setTabletTracking(True)
        self.label_Ni.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Ni.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Ni.setObjectName("label_Ni")
        self.verticalLayout.addWidget(self.label_Ni)
        self.label_Rb = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Rb.setFont(font)
        self.label_Rb.setTabletTracking(True)
        self.label_Rb.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Rb.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Rb.setObjectName("label_Rb")
        self.verticalLayout.addWidget(self.label_Rb)
        self.label_Sr = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Sr.setFont(font)
        self.label_Sr.setTabletTracking(True)
        self.label_Sr.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Sr.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Sr.setObjectName("label_Sr")
        self.verticalLayout.addWidget(self.label_Sr)
        self.label_Ti = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Ti.setFont(font)
        self.label_Ti.setTabletTracking(True)
        self.label_Ti.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Ti.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Ti.setObjectName("label_Ti")
        self.verticalLayout.addWidget(self.label_Ti)
        self.label_Zn = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_Zn.setFont(font)
        self.label_Zn.setTabletTracking(True)
        self.label_Zn.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_Zn.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.label_Zn.setObjectName("label_Zn")
        self.verticalLayout.addWidget(self.label_Zn)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.lineEdit_Al = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Al.setObjectName("lineEdit_Al")
        self.verticalLayout_2.addWidget(self.lineEdit_Al)
        self.lineEdit_Ba = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Ba.setObjectName("lineEdit_Ba")
        self.verticalLayout_2.addWidget(self.lineEdit_Ba)
        self.lineEdit_Ca = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Ca.setObjectName("lineEdit_Ca")
        self.verticalLayout_2.addWidget(self.lineEdit_Ca)
        self.lineEdit_Cu = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Cu.setObjectName("lineEdit_Cu")
        self.verticalLayout_2.addWidget(self.lineEdit_Cu)
        self.lineEdit_Fe = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Fe.setObjectName("lineEdit_Fe")
        self.verticalLayout_2.addWidget(self.lineEdit_Fe)
        self.lineEdit_K = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_K.setObjectName("lineEdit_K")
        self.verticalLayout_2.addWidget(self.lineEdit_K)
        self.lineEdit_Li = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Li.setObjectName("lineEdit_Li")
        self.verticalLayout_2.addWidget(self.lineEdit_Li)
        self.lineEdit_Mg = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Mg.setObjectName("lineEdit_Mg")
        self.verticalLayout_2.addWidget(self.lineEdit_Mg)
        self.lineEdit_Mn = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Mn.setObjectName("lineEdit_Mn")
        self.verticalLayout_2.addWidget(self.lineEdit_Mn)
        self.lineEdit_Na = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Na.setObjectName("lineEdit_Na")
        self.verticalLayout_2.addWidget(self.lineEdit_Na)
        self.lineEdit_Ni = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Ni.setObjectName("lineEdit_Ni")
        self.verticalLayout_2.addWidget(self.lineEdit_Ni)
        self.lineEdit_Rb = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Rb.setObjectName("lineEdit_Rb")
        self.verticalLayout_2.addWidget(self.lineEdit_Rb)
        self.lineEdit_Sr = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Sr.setObjectName("lineEdit_Sr")
        self.verticalLayout_2.addWidget(self.lineEdit_Sr)
        self.lineEdit_Ti = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Ti.setObjectName("lineEdit_Ti")
        self.verticalLayout_2.addWidget(self.lineEdit_Ti)
        self.lineEdit_Zn = QtWidgets.QLineEdit(self.horizontalLayoutWidget_2)
        self.lineEdit_Zn.setObjectName("lineEdit_Zn")
        self.verticalLayout_2.addWidget(self.lineEdit_Zn)
        self.horizontalLayout_4.addLayout(self.verticalLayout_2)
        self.label_input_head = QtWidgets.QLabel(self.centralwidget)
        self.label_input_head.setGeometry(QtCore.QRect(120, 90, 251, 16))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_input_head.setFont(font)
        self.label_input_head.setAlignment(QtCore.Qt.AlignCenter)
        self.label_input_head.setObjectName("label_input_head")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(119, 0, 251, 80))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_wt_head = QtWidgets.QLabel(self.verticalLayoutWidget_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_wt_head.setFont(font)
        self.label_wt_head.setAlignment(QtCore.Qt.AlignCenter)
        self.label_wt_head.setObjectName("label_wt_head")
        self.verticalLayout_3.addWidget(self.label_wt_head)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioButton_white = QtWidgets.QRadioButton(self.verticalLayoutWidget_3)
        self.radioButton_white.setChecked(True)
        self.radioButton_white.setObjectName("radioButton_white")
        self.horizontalLayout_2.addWidget(self.radioButton_white)
        self.radioButton_red = QtWidgets.QRadioButton(self.verticalLayoutWidget_3)
        self.radioButton_red.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.radioButton_red.setChecked(False)
        self.radioButton_red.setObjectName("radioButton_red")
        self.horizontalLayout_2.addWidget(self.radioButton_red)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.label_results = QtWidgets.QLabel(self.centralwidget)
        self.label_results.setGeometry(QtCore.QRect(420, 180, 561, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_results.setFont(font)
        self.label_results.setAlignment(QtCore.Qt.AlignCenter)
        self.label_results.setObjectName("label_results")
        self.label_results.setVisible(False)
        self.pushButton_calculate = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_calculate.setGeometry(QtCore.QRect(420, 530, 191, 28))
        self.pushButton_calculate.setObjectName("pushButton_calculate")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton_calculate.clicked.connect(self.calculate)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "Классификатор вин", None, -1))
        self.label_svm_head.setText(QtWidgets.QApplication.translate("MainWindow", "Метод опорных векторов:", None, -1))
        self.label_type_head.setText(QtWidgets.QApplication.translate("MainWindow", "Наименование", None, -1))
        self.label_knn_head.setText(
            QtWidgets.QApplication.translate("MainWindow", "Метод k-ближайших соседей:", None, -1))
        self.label_zone_head.setText(QtWidgets.QApplication.translate("MainWindow", "Зона", None, -1))
        self.label_nbc_head.setText(
            QtWidgets.QApplication.translate("MainWindow", "Наивный Байесовский классификатор:", None, -1))
        self.label_Al.setText(QtWidgets.QApplication.translate("MainWindow", "Al", None, -1))
        self.label_Ba.setText(QtWidgets.QApplication.translate("MainWindow", "Ba", None, -1))
        self.label_Ca.setText(QtWidgets.QApplication.translate("MainWindow", "Ca", None, -1))
        self.label_Cu.setText(QtWidgets.QApplication.translate("MainWindow", "Cu", None, -1))
        self.label_Fe.setText(QtWidgets.QApplication.translate("MainWindow", "Fe", None, -1))
        self.label_K.setText(QtWidgets.QApplication.translate("MainWindow", "K", None, -1))
        self.label_Li.setText(QtWidgets.QApplication.translate("MainWindow", "Li", None, -1))
        self.label_Mg.setText(QtWidgets.QApplication.translate("MainWindow", "Mg", None, -1))
        self.label_Mn.setText(QtWidgets.QApplication.translate("MainWindow", "Mn", None, -1))
        self.label_Na.setText(QtWidgets.QApplication.translate("MainWindow", "Na", None, -1))
        self.label_Ni.setText(QtWidgets.QApplication.translate("MainWindow", "Ni", None, -1))
        self.label_Rb.setText(QtWidgets.QApplication.translate("MainWindow", "Rb", None, -1))
        self.label_Sr.setText(QtWidgets.QApplication.translate("MainWindow", "Sr", None, -1))
        self.label_Ti.setText(QtWidgets.QApplication.translate("MainWindow", "Ti", None, -1))
        self.label_Zn.setText(QtWidgets.QApplication.translate("MainWindow", "Zn", None, -1))
        self.label_input_head.setText(QtWidgets.QApplication.translate("MainWindow", "Элементный состав", None, -1))
        self.label_wt_head.setText(QtWidgets.QApplication.translate("MainWindow", "Тип вина", None, -1))
        self.radioButton_white.setText(QtWidgets.QApplication.translate("MainWindow", "Белое", None, -1))
        self.radioButton_red.setText(QtWidgets.QApplication.translate("MainWindow", "Красное", None, -1))
        self.label_results.setText(QtWidgets.QApplication.translate("MainWindow", "Результаты", None, -1))
        self.pushButton_calculate.setText(QtWidgets.QApplication.translate("MainWindow", "Классифицировать", None, -1))

    @QtCore.Slot()
    def calculate(self):
        self.label_results.setVisible(True)
        elements = ['Al', 'Ba', 'Ca', 'Cu', 'Fe', 'K', 'Li', 'Mg', 'Mn', 'Na', 'Ni', 'Rb', 'Sr', 'Ti', 'Zn']
        element_values = list()
        for element in elements:
            element_values.append(float(convert_colons(eval('self.lineEdit_{}.text()'.format(element)))))
        element_values = np.array(element_values)
        if self.radioButton_white.isChecked():
            wine_type = 'white'
        else:
            wine_type = 'red'
        ret_dict = self.models.predict(wine_type, element_values)
        for method in ['svm', 'knn', 'nbc']:
            exec('self.label_{}_head.setVisible(True)'.format(method))
            for class_type in ['type', 'zone']:
                exec('self.label_{}_head.setVisible(True)'.format(class_type))
                current_value = ret_dict['{}_{}'.format(method, class_type)][0]
                exec('self.label_{}_{}.setText(current_value)'.format(method, class_type))
                exec('self.label_{}_{}.setVisible(True)'.format(method, class_type))