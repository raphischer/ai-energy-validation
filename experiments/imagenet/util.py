import subprocess
import platform
import re
import time

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)
    
def print_colored_block(message, ok=True, rows=6, row_length=80):
    col = '\033[92m' if ok else '\033[91m'
    print(col + (u"\u2588"*row_length + '\n')*rows + '\n')
    print(f"{message}\n")
    print((u"\u2588"*row_length + '\n')*rows + '\033[0m')

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode('ascii')
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1).strip()
    return ""

def save_webcam_image(fname, device_id=0, seconds=8):
    import cv2
    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
    print( f"Saving webcam image (device {device_id}), will take {seconds} seconds..." )

    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame")
            continue
        # Wait 1 second
        time.sleep(1)
    cv2.imwrite(fname, frame)
    cap.release()
