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
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.drowsiness = 0 # Initialize drowsiness meter
        
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
        self.drowsiness_list = [] # Stores list of drowsiness frames

        self.graph_status = True
        self.time_axis = None # Representation of time in the productivity graph
        self.productivity_axis = None # Productivity over time in the productivity graph
        self.productivity_val = 100
        self.final_person_count = 0
        self.final_cellphone_count = 0
        self.final_drowsy_count = 0
        sns.set_theme()
        sns.set_context("paper")

        widgets.startTimerButton.clicked.connect(self.start_timer) # Start Button
        widgets.stopTimerButton.clicked.connect(self.stop_timer) # Stop Button
        
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s') # YOLO model initialization
        self.model.classes = [0, 67] # Classify only person (0) and cellphone (67)

        self.model_2 = torch.hub.load('ultralytics/yolov5', 'custom', path='drowsiness.pt', force_reload=True) # Drowsiness model initialization


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
        widgets.btn_new.clicked.connect(self.buttonClick)


        self.show() # SHOW APP

        # SET CUSTOM THEME
        useCustomTheme = False
        themeFile = "themes\py_dracula_dark.qss"

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
    
    def updated_graph(self, productivity_val):
        fig = Figure(figsize = (5.31, 2.11))
        ax = fig.add_subplot()
        ax.set_xlabel('Time')
        ax.set_ylabel('Productivity Level')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        fig.set_facecolor('none')
        if self.time_axis != None:
            self.time_axis.append(self.time_axis[-1] + 1)
        else:
            self.time_axis = [1]
        if self.productivity_axis != None:
            self.productivity_axis.append(productivity_val)
        else:
            self.productivity_axis = [productivity_val]
        ax.plot(self.time_axis, self.productivity_axis, color='red')
        ax.set_facecolor('none')
        canvas = FigureCanvas(fig)
        
        widgets.productivityGraph.takeAt(0)
        widgets.productivityGraph.addWidget(canvas)
        
    def remove_graph(self):
        widgets.productivityGraph.takeAt(0)
        self.time_axis = None
        self.productivity_axis = None
	

    def start_timer(self):
        if self.timer_id:
            # If timer is already running, stop it first
            self.killTimer(self.timer_id)

        if (self.time_axis != None) or (self.productivity_axis != None):
            self.remove_graph()
        
        self.timer_start_time = cv2.getTickCount()  # Get current tick count
        self.timer_id = self.startTimer(self.timer_update_interval)  # Start the timer
        self.started = True

    def stop_timer(self):
        if self.timer_id:
            self.killTimer(self.timer_id)
            self.timer_id = None
            widgets.timerLabel.display("00:00:00")
            self.updated_graph(self.productivity_val)
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

                results2 = self.model_2(frame) # Process drowsiness
                pred2 = results2.pred[0]

                person_detected = False
                phone_detected = False
                drowsiness_detected = False

                # Get the status of the checkboxes
                person_detection_enabled = widgets.personLabel.isChecked()
                phone_detection_enabled = widgets.phoneLabel.isChecked()
                drowsiness_enabled = widgets.drowsinessLabel.isChecked()

                # Step 2: Draw bounding boxes and labels on the frame
                for det in pred:
                    if det[4] > 0.1:  # Confidence threshold, can be adjusted
                        x1, y1, x2, y2 = map(int, det[:4])
                        label_name = self.model.names[int(det[5])]
                                        
                        label = f'{label_name} {det[4]:.2f}'  

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
                                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw a green rectangle
                                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                for det in pred2:
                    if det[4] > 0.1:
                        x1, y1, x2, y2 = map(int, det[:4])
                        label_name = self.model_2.names[int(det[5])]
                                    
                        label = f'{label_name} {det[4]:.2f}'  

                        if label_name == 'drowsy':
                            drowsiness_detected = True
                        if drowsiness_enabled:
                                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw a green rectangle
                                frame = cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


                # Check if a person was detected and reset or increment counter
                if person_detected and person_detection_enabled:
                    if self.person_not_in_frame != 0:
                        self.person_not_in_frame_list.append(self.person_not_in_frame)
                    self.person_not_in_frame = 0
                else:
                    self.person_not_in_frame += 1
                    if person_detection_enabled:
                        self.productivity_val -= 5
                        self.final_person_count += 1


                # Check if a cell phone was detected and increment counter
                if phone_detected and phone_detection_enabled:
                    self.phone_in_frame += 1
                    if phone_detection_enabled:
                        self.productivity_val -= 5
                        self.final_cellphone_count += 1
                else:
                    if self.phone_in_frame != 0:
                        self.phone_in_frame_list.append(self.phone_in_frame)
                    self.phone_in_frame = 0

                # Check if drowsiness detected and incremeent counter
                if drowsiness_detected and drowsiness_enabled:
                    self.drowsiness += 1
                    if drowsiness_enabled:
                        self.productivity_val -= 5
                        self.final_drowsy_count += 1
                else:
                    if self.drowsiness != 0:
                        self.drowsiness_list.append(self.drowsiness)
                    self.drowsiness = 0

                # Check if cell phone has been in frame for more than 5 frames
                if (self.phone_in_frame > 5) and self.started:
                    self.play_alert_sound()
                    self.phone_in_frame = 0  # Resetting the count after playing the sound.

                # Check if a person has been away for more than 5 frames
                if (self.person_not_in_frame > 5) and self.started:
                    self.play_alert_sound()
                    self.person_not_in_frame = 0  # Resetting the count after playing the sound.

                # Check if a person has been drowsy for more than 5 frames
                if (self.drowsiness > 15) and self.started:
                    self.play_alert_sound()
                    self.drowsiness = 0

                # Step 3: Display the frame
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(533, 400, Qt.AspectRatioMode.KeepAspectRatio)
                widgets.videoLabel.setPixmap(QPixmap.fromImage(p))
        # Update Graph Here:
        if self.started:
            self.updated_graph(self.productivity_val)
            if self.productivity_val <= 100:
                self.productivity_val += 1


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

        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page) # SET PAGE
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU
            self.start_video_feed()

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
