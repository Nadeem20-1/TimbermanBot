# # Imports
# from ultralytics import YOLO
# import time
# import threading
# import numpy as np
# import cv2
# import pyautogui
# from mss import mss
# import mss.tools

# class TimbermanBot:
#     def __init__(self):
#         self.model = YOLO('/Users/nadeemee/PythonProjects/Project 1/OpenCV/runs/detect/train19/weights/best.pt')
#         self.model2 = YOLO('/Users/nadeemee/PythonProjects/Project 1/OpenCV/runs/detect/train19/weights/best_openvino_model/')
#         self.model3 = YOLO('/Users/nadeemee/PythonProjects/Project 1/OpenCV/runs/detect/train19/weights/best.mlpackage')

#         self.bounding_box = {'top': 80, 'left' : 900, 'width' : 900, 'height': 585}
#         self.bounding_box2 = {'top': 80, 'left' : 1100, 'width' : 500, 'height': 585}
        
#         self.timber_detected = False
#         self.game_stopped = False
#         self.direction = True
#         self.pushed = False

#     def bb_intersection_over_union(self, boxA, boxB):
#         # determine the (x, y)-coordinates of the intersection rectangle
#         xA = max(boxA[0], boxB[0])
#         yA = max(boxA[1], boxB[1])
#         xB = min(boxA[2], boxB[2])
#         yB = min(boxA[3], boxB[3])
#         # compute the area of intersection rectangle
#         interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
#         # compute the area of both the prediction and ground-truth
#         # rectangles
#         boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
#         boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
#         # compute the intersection over union by taking the intersection
#         # area and dividing it by the sum of prediction + ground-truth
#         # areas - the interesection area
#         iou = float(interArea) / float(boxAArea + boxBArea - interArea)
#         # return the intersection over union value
#         return iou

#     def model_runner(self):
#         while True:
#             sct = mss.mss()
#             img = sct.grab(self.bounding_box2)
#             img = np.array(img)
#             img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#             results = self.model3(img, verbose=False)
#             cv2.imshow('YOLO', results[0].plot())
            
#             for result in results:
#                 try:
#                     b = result.boxes.cls.tolist()
#                     b1 = result.boxes.xyxy.tolist()
#                     index = b.index(0)
#                     self.timber_detected = True
                    
#                     for r in b:
#                         if r != index:
#                             b1[index][1] -= 30
#                             res = self.bb_intersection_over_union(b1[index], b1[int(r)])
#                             # print(res)
#                             if res > 0 and self.pushed == False:
#                                 self.pushed = True
#                                 self.direction = not self.direction
#                 except:
#                     self.timber_detected = False
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         cv2.destroyAllWindows()
#         self.game_stopped = True

#     def game_runner(self):
#         pyautogui.PAUSE = 0.4
#         while self.game_stopped == False:
#             # if self.game_stopped == True:
#             #     break
#             if self.timber_detected == True:
#                 if self.direction == True:
#                     pyautogui.press('left')
#                     print("Left Pressed")
#                 else:
#                     pyautogui.press('right')
#                     print("Right Pressed")
#                 self.pushed = False
    
            

# if __name__ == '__main__':
#     game = TimbermanBot()
#     # game.model_runner()
    
#     thread_a = threading.Thread(target=game.model_runner)
#     thread_b = threading.Thread(target=game.game_runner)
    
#     thread_a.start()
#     thread_b.start()
    
#     thread_a.join()
#     thread_b.join()
    

# Using multiple processors

# Imports
from ultralytics import YOLO
import time
import multiprocessing
import numpy as np
import cv2
import pyautogui
from mss import mss
import mss.tools

class TimbermanBot:
    def __init__(self):
        self.model = YOLO('/Users/nadeemee/PythonProjects/Project 1/OpenCV/runs/detect/train19/weights/best.pt')
        self.model2 = YOLO('/Users/nadeemee/PythonProjects/Project 1/OpenCV/runs/detect/train19/weights/best_openvino_model/')
        self.model3 = YOLO('/Users/nadeemee/PythonProjects/Project 1/OpenCV/runs/detect/train19/weights/best.mlpackage')

        # self.bounding_box = {'top': 80, 'left' : 900, 'width' : 900, 'height': 585}
        self.bounding_box2 = {'top': 80, 'left' : 1100, 'width' : 500, 'height': 585}
        
        self.pushed_lock = multiprocessing.Lock()
        

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

    def screenshot(self):
        sct = mss.mss()
        img = sct.grab(self.bounding_box2)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def model_runner(self, timber_detected, game_stopped, direction, pushed):
        while True:
            img = self.screenshot()
            results = self.model3(img, verbose=False)
            cv2.imshow('YOLO', results[0].plot())
            
            try:
                b = results[0].boxes.cls.tolist()
                b1 = results[0].boxes.xyxy.tolist()
                index = b.index(0)
                timber_detected.value = True
                
                # Issue: Since it is looping through each one,
                # Need to make move AFTEr looping through all
                res = 0
                for r in b:
                    if r != index:
                        b1[index][1] -= 50
                        if res == 0:
                            res = self.bb_intersection_over_union(b1[index], b1[int(r)])
                print(res)
                if res > 0 and pushed.value == False:
                    print("1: ",direction.value)
                    print(b1[index][0], direction.value)
                    # Additional checks implemented to ensure timberman
                    # doesn't switch back if direction and position don't
                    # line up
                    if b1[index][0] > 500 and direction.value == False:
                        direction.value = not direction.value
                    elif b1[index][0] < 300 and direction.value == True:
                        direction.value = not direction.value
                    print("2: ",direction.value)
                    with self.pushed_lock:
                        if not pushed.value:
                            pushed.value = True
                    # direction.value = not direction.value
            except:
                timber_detected.value = False
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        game_stopped.value = True

    def game_runner(self, timber_detected, game_stopped, direction, pushed):
        pyautogui.PAUSE = 0.0
        while game_stopped.value == False:
            direction1 = direction.value
            if timber_detected.value == True:
                # with self.pushed_lock:
                
                if direction1 == True:
                    pyautogui.press('left')
                    print("Left Pressed")
                else:
                    pyautogui.press('right')
                    print("Right Pressed")
                    
                pushed.value = False
                time.sleep(0.1)
                
    def game_restart(self):
        pyautogui.press('space')

if __name__ == '__main__':
    game = TimbermanBot()
    # game.model_runner()
    
    manager = multiprocessing.Manager()
    
    timber_detected = manager.Value(bool, False)
    game_stopped = manager.Value(bool, False)
    direction = manager.Value(bool, True)
    pushed = manager.Value(bool, False)
    
    process_a = multiprocessing.Process(target=game.model_runner, args=(timber_detected, game_stopped, direction, pushed,))
    process_b = multiprocessing.Process(target=game.game_runner, args=(timber_detected, game_stopped, direction, pushed,))
    
    process_b.start()
    process_a.start()
    
    process_a.join()
    process_b.join()
    