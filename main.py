import sys
import os
import platform

import cv2
import torch
from ultralytics import YOLO
import pygame

from modules import *
from widgets import *

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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

        # Flag for when we start timer
        self.started = False
        
        # Webcam
        self.cap = None # Webcam off as default
        self.person_not_in_frame = 0 # Initialize counter for when person leaves
        self.phone_in_frame = 0 # Initialize counter for cell phone detection
        
        # Time
        self.minute = 0
        self.seconds = 0
        self.milliseconds = 0

        self.timer_start_time = None  # Time when the timer was started
        self.timer_update_interval = 1  # Update every second
        self.timer_id = None  # ID of the timer

        widgets.timerLabel.display("00:00:00")

        # Data Collection
        self.stopwatch_list = [] # Stores our list of stopwatch times
        self.person_not_in_frame_list = [] # Stores our list of person leaving frame frames
        self.phone_in_frame_list = [] # Stores our list of phone being in camera


        widgets.startTimerButton.clicked.connect(self.start_timer) # Start Button
        widgets.stopTimerButton.clicked.connect(self.stop_timer) # Stop Button
        
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # YOLO model initialization
        self.model.classes = [0, 67] # Classify only person (0) and cellphone (67)

        # App Settings
        Settings.ENABLE_CUSTOM_TITLE_BAR = True
        title = "FocusGuardian"
        description = "FocusGuardian - AI driven solutions to help you ace more exams."
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True)) # TOGGLE MENU

        UIFunctions.uiDefinitions(self) # SET UI DEFINITIONS

        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch) # QTableWidget PARAMETERS


        ### BUTTONS CLICK ###

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

        self.show() # SHOW APP

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

        # CUSTOM DESIGN FOR CHECKS
        checkbox_stylesheet = """
            QCheckBox::indicator {
                border: 2px solid #76acdb;
                width: 15px;
                height: 15px;
                border-radius: 7px;
                background-color: transparent;
            }
            QCheckBox::indicator:checked {
                image: url(none.png);
                border: 2px solid #76acdb;
                background-color: #222);
            }
            """

        widgets.personLabel.setStyleSheet(checkbox_stylesheet)
        widgets.phoneLabel.setStyleSheet(checkbox_stylesheet)
        widgets.drowsinessLabel.setStyleSheet(checkbox_stylesheet)

    def play_alert_sound(self):
        pygame.mixer.init()
        pygame.mixer.music.load('radar.mp3')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

    def start_timer(self):
        if self.timer_id:
            # If timer is already running, stop it first
            self.killTimer(self.timer_id)
        
        self.timer_start_time = cv2.getTickCount()  # Get current tick count
        self.timer_id = self.startTimer(self.timer_update_interval)  # Start the timer
        self.started = True

    def stop_timer(self):
        if self.timer_id:
            self.killTimer(self.timer_id)
            self.timer_id = None
            widgets.timerLabel.display("00:00:00")

            self.stopwatch_list.append((self.minutes, self.seconds, self.milliseconds))
            self.minutes, self.seconds, self.milliseconds = 0, 0, 0
            self.started = False
        
    def start_video_feed(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            
        # Timer to update the video feed
        self.timer = self.startTimer(42)  # Update every 30ms

    def stop_video_feed(self):
        if self.cap:
            self.killTimer(self.timer)
            self.cap.release()
            self.cap = None

    def timerEvent(self, event):
        if event.timerId() == self.timer_id:
            # Timer event for our timer
            elapsed_ticks = cv2.getTickCount() - self.timer_start_time
            elapsed_time = elapsed_ticks / cv2.getTickFrequency() # Convert to seconds
            self.minutes, self.seconds = divmod(elapsed_time, 60)
            _, self.milliseconds = divmod(elapsed_time * 1000, 1000)
            widgets.timerLabel.display(f"{int(self.minutes):02d}:{int(self.seconds):02d}:{int(self.milliseconds):02d}")
        else:
            ret, frame = self.cap.read()
            if ret:
                # Step 1: Process frame with YOLOv5
                results = self.model(frame)  # Process the frame
                pred = results.pred[0] # Get the first prediction (in case of batch processing)

                person_detected = False
                phone_detected = False

                # Get the status of the checkboxes
                person_detection_enabled = widgets.personLabel.isChecked()
                phone_detection_enabled = widgets.phoneLabel.isChecked()

                # Step 2: Draw bounding boxes and labels on the frame
                for det in pred:
                    if det[4] > 0.1:  # Confidence threshold, can be adjusted
                        x1, y1, x2, y2 = map(int, det[:4])
                        label_name = self.model.names[int(det[5])]
                                        
                        label = f'{label_name} {det[4]:.2f}'  # Move this line up here

                        if label_name == "person":
                            person_detected = True
                            # Only draw if the checkbox is checked
                            if person_detection_enabled:
                                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle
                                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif label_name == "cell phone":
                            phone_detected = True
                            # Only draw if the checkbox is checked
                            if phone_detection_enabled:
                                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle
                                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Check if a person was detected and reset or increment counter
                if person_detected and person_detection_enabled:
                    if self.person_not_in_frame != 0:
                        self.person_not_in_frame_list.append(self.person_not_in_frame)
                    self.person_not_in_frame = 0
                else:
                    self.person_not_in_frame += 1

                # Check if a cell phone was detected and increment counter
                if phone_detected and phone_detection_enabled:
                    self.phone_in_frame += 1
                else:
                    if self.phone_in_frame != 0:
                        self.phone_in_frame_list.append(self.phone_in_frame)
                    self.phone_in_frame = 0

                # Check if cell phone has been in frame for more than 5 frames
                if (self.phone_in_frame > 5) and self.started:
                    self.play_alert_sound()
                    self.phone_in_frame = 0  # Resetting the count after playing the sound.

                # Check if a person has been away for more than 5 frames
                if (self.person_not_in_frame > 5) and self.started:
                    self.play_alert_sound()
                    self.person_not_in_frame = 0  # Resetting the count after playing the sound.

                # Step 3: Display the frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(533, 400, Qt.AspectRatioMode.KeepAspectRatio)
                widgets.videoLabel.setPixmap(QPixmap.fromImage(p))

    # BUTTONS CLICK
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            self.stop_video_feed()

        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
            self.stop_video_feed()

        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page) # SET PAGE
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU
            self.start_video_feed()
            x_axis = [1, 2, 8, 3, 6]
            y_axis = [9, 3, 1, 6, 3]

            fig = Figure(figsize = (6, 6))
            ax = fig.add_subplot()
            ax.plot(x_axis, y_axis)

            canvas = FigureCanvas(fig)
            widgets.productivityGraph.addWidget(canvas)

        print(f'Button "{btnName}" pressed!')


    # RESIZE EVENTS
    def resizeEvent(self, event):
        UIFunctions.resize_grips(self) # Update Size Grips

    # MOUSE CLICK EVENTS
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos() # SET DRAG POS WINDOW

        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
            
            print(f"User's list where person leaves frame: {self.person_not_in_frame_list}")
            print(f"User's list of stopwatch times, stored in (minutes, seconds, milliseconds): {self.stopwatch_list}")
            print(f"User's list of frames where cell phone is detected: {self.phone_in_frame_list}")

        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())
