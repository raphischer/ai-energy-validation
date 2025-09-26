import time
import cv2

def init_webcam(device_id=0, seconds=10):
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    print( f"Initializing webcam on device {device_id} for {seconds} seconds..." )

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame")
            continue
        cv2.imwrite('webcam_test.jpg', frame)
        # Wait 1 second
        time.sleep(1)

    cap.release()
    print("✅ Webcam test completed, image saved as 'webcam_test.jpg'")

init_webcam()