import argparse
import os

import cv2
import pytesseract

def draw_rectangle(event, x, y, flags, param):
    global roi, drawing, start_point
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)  # Store the starting point of the rectangle

    elif event == cv2.EVENT_MOUSEMOVE:
        try:
            if drawing:
                # Temporary rectangle as the user drags the mouse
                frame_copy = param.copy()
                cv2.rectangle(frame_copy, start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Select ROI", frame_copy)
        except NameError:
            pass

def select_roi(frame):
    # Display the frame and set the mouse callback
    cv2.imshow("Select ROI", frame)
    cv2.setMouseCallback("Select ROI", draw_rectangle, frame)
    print("Draw a rectangle to select the region of interest (ROI).")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # make sure that the coordinates are in correct ordner, no matter how the user draws the rectangle
    x1, y1, x2, y2 = roi
    if x1 > x2:
        fr, x1 = x1, x2
        x2 = fr
    if y1 > y2:
        fr, y1 = y1, y2
        y2 = fr
    return x1, y1, x2, y2

os.environ['TESSDATA_PREFIX'] = r'/home/fischer/repos/digital-display-character-rec/letsgodigital'

roi = select_roi(cv2.imread(os.path.join(output_folder, frame_names[0])))