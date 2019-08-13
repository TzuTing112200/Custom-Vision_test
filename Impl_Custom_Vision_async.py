import cv2
import threading
import numpy as np
import matplotlib.pyplot as plt

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

ENDPOINT = "https://westus2.api.cognitive.microsoft.com"
prediction_key = "8c920d2f81f04ab7bccac7176a4cfd21"
predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

results = 0

def detect():
    global results
    
    mutex.acquire()
    with open("img.jpg", mode="rb") as test_data:
        results = predictor.detect_image("a15f8db0-1845-4715-83ba-6b7b763726e9", "MS_Ocean_World_1080716", test_data)
    mutex.release()

mutex = threading.Lock()

def main():
    global results
    
    cap = cv2.VideoCapture(0)
    '''
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    '''
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    ret, img = cap.read()
    cv2.imwrite("img.jpg", img)
    for i in range(1000):
        
        results += 10
    t = threading.Thread(target = detect)
    t.start()
    t.join()

    while(cap.isOpened()):
        ret, img = cap.read()
        cv2.imwrite("img.jpg", img)

        if not t.is_alive():
            t = threading.Thread(target = detect)
            t.start()
        
        for prediction in results.predictions:
            if prediction.probability * 100 >= 60:
                img = img.copy()
                width = img.shape[1]
                height = img.shape[0]
                new_lefttop_x = int(prediction.bounding_box.left * width)
                new_lefttop_y = int(prediction.bounding_box.top * height)
                new_rightbot_x = int(new_lefttop_x + (prediction.bounding_box.width * width))
                new_rightbot_y = int(new_lefttop_y + (prediction.bounding_box.height * height))
                cv2.rectangle(img, (new_lefttop_x, new_lefttop_y), (new_rightbot_x, new_rightbot_y), (0, 0, 256), 7)
                text_lefttop_y = int(new_lefttop_y - height * 0.05)
                if text_lefttop_y < 50:
                    text_lefttop_y = int(new_rightbot_y + height * 0.12)
                cv2.putText(img, prediction.tag_name, (new_lefttop_x, text_lefttop_y),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 256), 1, cv2.LINE_AA)
    
        cv2.imshow("image", img)
    
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break

    t.join()
    cap.release()
    cv2.destroyAllWindows()
    
main()
