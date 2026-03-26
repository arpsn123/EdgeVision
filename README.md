# EdgeVision

## A Distributed IoT-Based Real-Time Object Detection and Tracking System Using YOLOv8 and Mobile Camera Nodes

---

## PROJECT OVERVIEW

This project presents the design and implementation of a distributed real-time computer vision system that integrates edge sensing devices with centralized machine learning inference. The system leverages a mobile device as an IoT-enabled camera node, which streams live video over a local network, and a compute node (laptop/desktop) that performs real-time object detection, tracking, and analytics using YOLOv8.

Unlike traditional standalone object detection systems that operate on static images or pre-recorded video, this system demonstrates a continuous, low-latency, network-driven inference pipeline. It mimics real-world deployments such as intelligent surveillance systems, smart traffic monitoring, and edge AI applications.

The core objective of this project is to bridge the gap between computer vision models and real-world system design by introducing networking, streaming protocols, and distributed processing into the pipeline.

---

## SYSTEM ARCHITECTURE

The system follows a distributed architecture consisting of three primary components:

1. Edge Sensor Node (Mobile Device)
2. Communication Layer (WiFi Network using IP-based streaming)
3. Compute Node (YOLOv8 Inference Engine on Laptop/Desktop)

**Data Flow:**

```
Mobile Camera → Frame Capture → Encoding → HTTP Stream  
→ Network Transmission → OpenCV Ingestion → YOLOv8 Inference  
→ Tracking + Analytics → Visualization + Storage  
```

This architecture reflects a simplified version of real-world edge AI systems where sensing and computation are decoupled.

---

## THEORETICAL FOUNDATIONS

### 1. Computer Vision Pipeline

The system operates on a frame-by-frame processing paradigm. Each frame extracted from the live stream undergoes:

* Preprocessing (color space handling via OpenCV)
* Inference using a convolutional neural network (YOLOv8)
* Post-processing (bounding boxes, class labels, tracking IDs)

This pipeline aligns with standard real-time vision systems where latency and throughput are critical.

---

### 2. Object Detection using YOLOv8

The system utilizes the YOLOv8 architecture provided by Ultralytics.

YOLO (You Only Look Once) is a single-stage object detection model that performs detection as a regression problem. Instead of generating region proposals (as in R-CNN), YOLO predicts bounding boxes and class probabilities directly from the image in a single forward pass.

**Key properties:**

* Real-time performance
* End-to-end differentiable architecture
* Grid-based detection mechanism
* Anchor-free detection (in newer versions like YOLOv8)

---

### 3. Object Tracking

Tracking is implemented using YOLOv8’s built-in tracking pipeline (ByteTrack-like approach).

Conceptually, tracking solves the problem:

> “How do we maintain identity of objects across frames?”

The system assigns unique IDs to detected objects and maintains them across consecutive frames using motion and appearance cues.

This enables:

* Unique object counting
* Avoiding duplicate detections
* Temporal understanding of motion

---

### 4. Real-Time Constraints

Real-time processing requires:

* Low latency frame acquisition
* Fast inference (model selection like yolov8n)
* Efficient rendering and I/O

The system balances accuracy and speed by using lightweight models and optimized frame handling.

---

## DEDICATED SECTION: MOBILE DEVICE AS AN IoT CAMERA NODE

This is the most critical conceptual contribution of the project.

---

### 1. Conceptual Understanding

The mobile device is transformed from a passive camera into an active networked sensor node.

Instead of storing or locally processing video, the device performs:

* Continuous frame capture
* Compression (JPEG encoding)
* Network streaming via HTTP

This effectively converts the phone into a lightweight video server.

---

### 2. Role of IP Webcam

The application functions as an HTTP server running on the mobile device. It exposes camera frames as a network-accessible stream.

Each device on a network is assigned an IP address (e.g., 192.168.x.x). The mobile device binds a server to this address and a port (typically 8080).

**Example endpoint:**

```
http://192.168.x.x:8080/video
```

When accessed, this endpoint continuously sends image frames encoded as MJPEG.

---

### 3. MJPEG Streaming

The stream is not a traditional video file. Instead, it is a sequence of JPEG images transmitted over HTTP.

**Structure:**

```
[Frame1][Frame2][Frame3]...
```

Each frame is individually encoded and sent, creating the illusion of motion when rendered sequentially.

**Advantages:**

* Simplicity
* Compatibility with OpenCV

**Limitations:**

* Higher bandwidth usage
* Less efficient than modern codecs

---

### 4. Network Communication

The system relies on local WiFi networking.

* The router assigns IP addresses to devices
* The laptop acts as a client
* The phone acts as a server

Communication occurs via HTTP requests and continuous data streaming.

This is a classic client-server model.

---

### 5. Integration with OpenCV

OpenCV provides a unified interface for video capture from both local devices and network streams.

By passing the stream URL:

```python
cap = cv2.VideoCapture("http://PHONE_IP:8080/video")
```

OpenCV internally:

* Sends HTTP requests
* Reads byte stream
* Decodes JPEG frames
* Returns frames as NumPy arrays

---

### 6. Why This is IoT

This setup qualifies as an IoT system because:

* A physical device (phone) senses the environment
* Data is transmitted over a network
* Another system processes and analyzes the data

This aligns with the fundamental definition of IoT-enabled sensing and distributed computation.

---

## IMPLEMENTATION DETAILS

Below is the core pipeline integrating streaming, detection, tracking, and saving.

---

### Core Code

```python
from ultralytics import YOLO  
import cv2  

model = YOLO("yolov8n.pt")  

url = "http://PHONE_IP:8080/video"  
cap = cv2.VideoCapture(url)  

ret, frame = cap.read()  
height, width = frame.shape[:2]  

fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out_raw = cv2.VideoWriter('raw_feed.avi', fourcc, 20.0, (width, height))  
out_annotated = cv2.VideoWriter('annotated_feed.avi', fourcc, 20.0, (width, height))  

seen_ids = set()  

while True:  
    ret, frame = cap.read()  
    if not ret:  
        break  

    out_raw.write(frame)  

    results = model.track(frame, persist=True)  
    annotated = results[0].plot()  

    boxes = results[0].boxes  
    if boxes is not None and boxes.id is not None:  
        for obj_id in boxes.id:  
            seen_ids.add(int(obj_id))  

    cv2.putText(annotated, f"Count: {len(seen_ids)}", (20,40),  
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)  

    out_annotated.write(annotated)  

    cv2.imshow("Tracking", annotated)  

    if cv2.waitKey(1) == 27:  
        break  

cap.release()  
out_raw.release()  
out_annotated.release()  
cv2.destroyAllWindows()  
```

---

## KEY FEATURES

* Real-time video ingestion from mobile device
* Distributed system design (sensor + compute separation)
* YOLOv8-based object detection
* Multi-object tracking with persistent IDs
* Unique object counting
* Dual video recording (raw + annotated)
* Scalable architecture for multi-camera extension

---

## APPLICATIONS

* Smart surveillance systems
* Crowd monitoring
* Traffic analysis
* Retail analytics (footfall counting)
* Edge AI deployments

---

## LIMITATIONS

* MJPEG streaming is bandwidth intensive
* Dependent on network stability
* Limited by CPU/GPU of compute node
* No hardware acceleration on mobile side

---

## 🧠 CONCLUSION

This project demonstrates how modern computer vision models can be integrated into real-world systems through distributed design principles. By transforming a mobile device into an IoT camera node and coupling it with a centralized inference engine, the system showcases a practical approach to real-time AI deployment.

It highlights the importance of combining machine learning with networking, systems engineering, and real-time processing to build scalable and impactful solutions.

Raw Feed : 
https://github.com/user-attachments/assets/d41275ec-e6f9-4c0c-8303-901a93ecfdc1

Annotated Feed :
https://github.com/user-attachments/assets/2a7f41b2-80b6-483b-87ce-3deba11bdb86


---
