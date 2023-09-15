import cv2
import numpy as np
from openvino import runtime as ov
import serial
import time

ie_core = ov.Core()
model = ie_core.read_model(model='model/ssdlite_mobilenet_v2_fp16.xml')

compiled_model = ie_core.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

height, width = list(input_layer.shape)[1:3]

classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "hair brush"
]
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

cap=cv2.VideoCapture(0)

py_serial=serial.Serial(port="COM3", baudrate=9600)

while True:
    ret,frame=cap.read()
    if ret:
        input_img = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
        input_img = input_img[np.newaxis, ...]
        results = compiled_model([input_img])[output_layer]
        h, w = frame.shape[:2]
        results = results.squeeze()
        n=0
        for _, label, score, xmin, ymin, xmax, ymax in results:
            if _==-1:
                break
            x1,y1,x2,y2=map(int,(xmin * w, ymin * h, xmax * w, ymax * h))
            label=int(label)
            if label==1:
                n=n+1
                score=float(score)
                color = tuple(map(int, colors[label]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame,f"{classes[label]} {score:.2f}",
                    (x1+10, y1+30),cv2.FONT_HERSHEY_COMPLEX,1,color,1,cv2.LINE_AA)
        if n>0:
            py_serial.write('1'.encode())
        elif n<1:
            py_serial.write('0'.encode())

        cv2.putText(frame,f"person : {n}",
            (10, 30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1,cv2.LINE_AA)
        cv2.imshow('cam',frame)
        if cv2.waitKey(33) ==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()