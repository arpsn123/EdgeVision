# %%
from ultralytics import YOLO
import cv2

print(cv2.__version__)
print(
    hasattr(cv2, "VideoCapture")
)  # VideoCapture ---> core OpenCV class for Webcam input, Video files, IP camera streams


# %%
# this is real time video processing loop using LIVE FEED

cap = cv2.VideoCapture(
    0
)  # 0 = default webcam, OpenCV internally uses OS camera drivers

if not cap.isOpened():  # camera/stream opened successfully or not;
    print("Camera not opened")
    exit()

while True:  # infinite loop --> continuously process video frames in real time

    # captures one frame from video stream in 'frame' --> image (NumPy array), 'ret' --> True/False (success)
    ret, frame = cap.read()

    if not ret:
        print("Frame not received")
        break

    # opens a GUI NATIVE Window and shows the current frame, this is real time rendering;
    cv2.imshow("Laptop Camera", frame)

    if (
        cv2.waitKey(1) == 27
    ):  # if press esc(27), then hold for 1milisec and then close(break); waitKey(1)---> without this the windows freezes;
        break

cap.release()  # releases camera resource
cv2.destroyAllWindows()  # ends all OpenCV windows

# %%
# same thing just the live webcam feed replaced with a IOT Device LIVE IP FEED

url = "http://192.168.29.83:8080/video"
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Cannot open stream")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received")
        break
    cv2.imshow("Phone Camera", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break
cap.release()
cv2.destroyAllWindows()  # %%


# %%

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    print("Cannot open stream")
    exit()

ret, frame = cap.read()
if not ret:
    print("Cannot read initial frame")
    exit()

height, width, _ = frame.shape


fourcc = cv2.VideoWriter_fourcc(
    *"XVID"
)  # fourcc ---> Four Character Code, tells OpenCV : how should i compress and store the video ????
# XVID ---> a codec = compression format, for .avi, here the 4 character is : X, V, I, D
# tells OpenCV how to encode the video while saving

out_raw = cv2.VideoWriter(
    "raw_feed.avi", fourcc, 20.0, (width, height)
)  # out_raw = saves original/raw frames; VideoWriter ---> to write a OpenCV frames in a video-file;
# "raw_feed.avi" ---> filename
# fourcc ---> codec
# 20.0 ---> fps
# (width, height) ---> resolution

out_annotated = cv2.VideoWriter("annotated_feed.avi", fourcc, 20.0, (width, height))

# the above 2 are not storing any feed(raw or annotated) right now, they r just output sink;

seen_ids = set()  # an empty set to store object IDs, set because no repetation;

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out_raw.write(
        frame
    )  # finally saving the  raw frames in previously defined output sink

    results = model.track(
        frame, persist=True
    )  # object detection + tracking on the current frame;
    # persist=True ---> Keeps memory of objects between frames, ie, same object = same ID across frames working little bit similarly to BYTETRACK

    annotated = results[0].plot()  # results[0] ---> first frame result

    boxes = results[
        0
    ].boxes  # extracting bounding boxes and their properties like : Class IDs, Confidence, Tracking IDs (boxes.id)

    if (
        boxes is not None and boxes.id is not None
    ):  # checking edgecase when some objects/boxes has no id,class etc(basically failed to detect); otherwise runtime error;

        for obj_id in boxes.id:
            seen_ids.add(int(obj_id))
            # Loop through all detected objects, get their tracking_IDs(obj_id), store in set(seen_ids)

    # just like showing PC GAMING STATS using MSI AFTER-BURNER, a overlay of count of unique detected objects;
    cv2.putText(
        annotated,  # the image
        f"Count: {len(seen_ids)}",  # the number of count and the label
        (20, 40),  # the position
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        1,  # size
        (0, 255, 0),  # green color
        2,  # thickness
    )

    out_annotated.write(
        annotated
    )  # finally saving the  annoated frames in previously defined output sink

    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out_raw.release()
out_annotated.release()
cv2.destroyAllWindows()
# %%
