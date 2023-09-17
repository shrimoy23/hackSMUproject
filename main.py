import sys
import os
import platform
import cv2
import torch
from ultralytics import YOLO

from modules import *
from widgets import *
os.environ["QT_FONT_DPI"] = "96" # FIX Problem for High DPI and Scale above 100%

# SET AS GLOBAL WIDGETS
widgets = None

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # Webcam off as default
        self.cap = None

        # Initialize timer variables
        self.timer_start_time = None  # Time when the timer was started
        self.timer_update_interval = 1000  # Update every second
        self.timer_id = None  # ID of the timer

        # Connect start timer button to method
        widgets.startTimerButton.clicked.connect(self.start_timer)
        widgets.stopTimerButton.clicked.connect(self.stop_timer)
        # YOLO model initialization
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        title = "PyDracula - Modern GUI"
        description = "PyDracula APP - Theme with colors based on Dracula for Python."
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        UIFunctions.uiDefinitions(self)

        # QTableWidget PARAMETERS
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # BUTTONS CLICK

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)

        # EXTRA LEFT BOX
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)
        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # SHOW APP
        self.show()

        # SET CUSTOM THEME
        useCustomTheme = False
        themeFile = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

    def start_timer(self):
        if self.timer_id:
            # If timer is already running, stop it first
            self.killTimer(self.timer_id)
        
        self.timer_start_time = cv2.getTickCount()  # Get current tick count
        self.timer_id = self.startTimer(self.timer_update_interval)  # Start the timer

    def stop_timer(self):
        if self.timer_id:
            self.killTimer(self.timer_id)
            self.timer_id = None
            widgets.timerLabel.setText("00:00")
        

    # Webcam Functionality
    def start_video_feed(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            
        # Timer to update the video feed
        self.timer = self.startTimer(30)  # Update every 30ms

    def stop_video_feed(self):
        if self.cap:
            self.killTimer(self.timer)
            self.cap.release()
            self.cap = None

    def timerEvent(self, event):
        if event.timerId() == self.timer_id:
            # Timer event for our timer
            elapsed_ticks = cv2.getTickCount() - self.timer_start_time
            elapsed_time = elapsed_ticks / cv2.getTickFrequency()  # Convert to seconds
            minutes, seconds = divmod(elapsed_time, 60)
            widgets.timerLabel.setText(f"{int(minutes):02d}:{int(seconds):02d}")
        else:
            ret, frame = self.cap.read()
            if ret:
                # Step 1: Process frame with YOLOv5
                results = self.model(frame)  # Process the frame
                pred = results.pred[0]       # Get the first prediction (in case of batch processing)

                # Step 2: Draw bounding boxes and labels on the frame
                for det in pred:
                    if det[4] > 0.3:  # Confidence threshold, can be adjusted
                        x1, y1, x2, y2 = map(int, det[:4])
                        label = f'{self.model.names[int(det[5])]} {det[4]:.2f}'
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle
                        frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Step 3: Display the frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
                widgets.label.setPixmap(QPixmap.fromImage(p))

    # BUTTONS CLICK
    # Post here your functions for clicked buttons
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            self.stop_video_feed()

        # SHOW WIDGETS PAGE
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            self.stop_video_feed()

        # SHOW NEW PAGE
        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page) # SET PAGE
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU
            self.start_video_feed()

        # PRINT BTN NAME
        print(f'Button "{btnName}" pressed!')


    # RESIZE EVENTS
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec_())
