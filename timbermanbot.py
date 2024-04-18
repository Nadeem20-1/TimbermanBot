# Imports
from ultralytics import YOLO
import time
import numpy as np
import cv2
import pyautogui
from mss import mss
import mss.tools
import pygetwindow
# traceback used for exception handling
import traceback

class TimbermanBot:
    def __init__(self):
        # Added different models. Using model3 as it is optimized for apple devices
        # Model1 is the original
        self.model1 = YOLO('detect/train19/weights/best.pt')
        # If you are using this model, ensure you: pip install openvino
        self.model2 = YOLO('detect/train19/weights/best_openvino_model/')
        # If you are using this model, ensure you: pip install coremltools
        self.model3 = YOLO('detect/train19/weights/best.mlpackage')

        self.timber_detected = False
        self.interception = False
        self.game_stopped = False
        self.direction = True
        self.message = False
        
        pyautogui.PAUSE = 0.0
        self.key_delay = 0.1
    
    # Grabbed this IOU function online. There are a couple ways
    # to do so, but they produce similar results.
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = float(interArea) / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def model_runner(self):
        while True:
            # Wrapped in try/except as timberman disappears when game-over is shown
            try:
                # Getting active window
                win = pygetwindow.getActiveWindow()
                window_dim = pygetwindow.getWindowGeometry(win)
                # If not timberman, don't proceed
                if win == 'Timberman Timberman':
                    self.message = False
                    bounding_box = {'top': window_dim[1]+20, 'left' : window_dim[0]+150, 'width' : window_dim[2]-300, 'height': window_dim[3]-20}
                    
                    middle = (window_dim[2]) / 2
                    print(window_dim)
                    print(window_dim[2], window_dim[0])
                    print(middle)
                    sct = mss.mss()
                    img = sct.grab(bounding_box)
                    img = np.array(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    results = self.model3(img, verbose=False)
                    cv2.imshow('YOLO', results[0].plot())
                    
                    b = results[0].boxes.cls.tolist()
                    b1 = results[0].boxes.xyxy.tolist()
                    # Timber index is 0, so if found, great! Otherwise -> except!
                    index = b.index(0)
                    self.timber_detected = True
                    # Setting interception value to 0, looping through
                    # list to see if any object is intercepting timberman
                    # (excluding timberman himself lol)
                    
                    # Adding offset for early detection, as well as I made my
                    # model too accurate in terms of the hitboxes (not enough room
                    # for errors, whoops)
                    b1[index][1] -= 60
                    for i, r in enumerate(b):
                        # Skipping over timberman
                        if i != index and r != 0:
                            # Getting the interception value, from 0 to 1
                            res = self.bb_intersection_over_union(b1[index], b1[i])
                            # If it's above 0, there's an interception!
                            if res > 0:
                                # Preventing occurences where it screenshots timberman multiple
                                # times, as it takes a bit of time to go left/right
                                if b1[index][0] > middle and self.direction == False or b1[index][0] < middle and self.direction == True:
                                    self.interception = True
                                break
                            else:
                                self.interception = False
                    self.game_runner(self.interception)
                elif self.message == False:
                    self.message = True
                    print("Timberman game is not active.")
            except:
                self.timber_detected = False
            # except Exception as e:
            #     print(e)
            #     traceback.print_exc()
            #     break
            #     self.timber_detected = False
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        self.game_stopped = True

    def game_runner(self, interception):
        # If interception, change directions
        if interception == True:
            self.direction = not self.direction
            
        # True is left, false is right
        if self.direction == True:
            pyautogui.press('left')
            # print("Left Pressed")
        else:
            pyautogui.press('right')
            # print("Right Pressed")   
        # Added a delay to give enough time for program to screenshot after
        # key input
        time.sleep(self.key_delay)

if __name__ == '__main__':
    game = TimbermanBot()
    game.model_runner()