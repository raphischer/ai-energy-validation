import argparse
import os
import warnings
import time
import shutil

from tqdm import tqdm
import numpy as np
import pandas as pd
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

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        roi = (*start_point, *end_point)
        frame_copy = param.copy()
        cv2.rectangle(frame_copy, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)

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

def apply_preprocessing(image, block_size, c_value, kernel_size, erosion_iterations):
    # Ensure block_size is odd and greater than 1
    if block_size % 2 == 0:
        block_size += 1
    # resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5) # resizing makes everything faster -> TODO move outside of this function
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value) # threshold
    kernel = np.ones((kernel_size, kernel_size), np.uint8) # create erosion & dilation kernel
    dilated_image = cv2.dilate(thresh, kernel, iterations=erosion_iterations)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_iterations)

    # kernel = np.ones((1, 1), np.uint8)
    # prep = cv2.erode(cv2.dilate(cv2.adaptiveThreshold(cv2.cvtColor(cv2.resize(test_frames[4], (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10), kernel, iterations=1), kernel, iterations=1)
    # cv2.imshow('test', prep)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return eroded_image

def display_with_text(images, texts, text_box_width=400):
    stacked = []
    for image, text in zip(images, texts):        
        # Write text on white canvas
        canvas = np.ones((image.shape[0], text_box_width, 3), dtype=np.uint8) * 255
        font, font_scale, thickness, line_height, y0 = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1, 25, 30
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * line_height
            cv2.putText(canvas, line, (10, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        # Concatenate the image with the text box horizontally
        stacked.append( np.hstack((image, canvas)) )
    # Display the vertically stacked images with text areas
    cv2.imshow('Preprocessing and OCR', np.vstack(stacked))

def update_preprocessing(x, get_ocr):
    try: # Get current positions of trackbars
        block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing and OCR')
        c_value = cv2.getTrackbarPos('C Value', 'Preprocessing and OCR')
        kernel_size = cv2.getTrackbarPos('Kernel Size', 'Preprocessing and OCR')
        erosion_iterations = cv2.getTrackbarPos('Erosion Iterations', 'Preprocessing and OCR')
    except Exception:
        return # only happen during initialization of window

    ocr, imgs = [], []
    for frame in frames:
        # Apply preprocessing with current parameters and store the OCR results
        processed_image = apply_preprocessing(frame, block_size, c_value, kernel_size, erosion_iterations)
        ocr.append( get_ocr(processed_image) )
        imgs.append( cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR) )

    # Display the processed image with the OCR text side by side
    display_with_text(imgs, ocr)

def interactive_preprocessing_with_ocr(images, get_ocr):
    global frames
    frames = images.copy()  # Store the image globally for access inside trackbar callback

    # Create window and trackbars for adjusting preprocessing parameters
    cv2.namedWindow('Preprocessing and OCR')
    cv2.createTrackbar('Block Size', 'Preprocessing and OCR', 21, 50, lambda x: update_preprocessing(x, get_ocr))
    cv2.createTrackbar('C Value', 'Preprocessing and OCR', 10, 20, lambda x: update_preprocessing(x, get_ocr))
    cv2.createTrackbar('Kernel Size', 'Preprocessing and OCR', 1, 20, lambda x: update_preprocessing(x, get_ocr))
    cv2.createTrackbar('Erosion Iterations', 'Preprocessing and OCR', 1, 10, lambda x: update_preprocessing(x, get_ocr))
    update_preprocessing(0, get_ocr) # call once for initial display

    # Keep the window open until the user presses 'Esc'
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
            break

    block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing and OCR')
    c_value = cv2.getTrackbarPos('C Value', 'Preprocessing and OCR')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Preprocessing and OCR')
    erosion_iterations = cv2.getTrackbarPos('Erosion Iterations', 'Preprocessing and OCR')
    cv2.destroyAllWindows()
    prep_params = (block_size, c_value, kernel_size, erosion_iterations)
    return lambda im: apply_preprocessing(im, block_size, c_value, kernel_size, erosion_iterations), prep_params

def detect_ocr(single_frame, ocr_func, preprocessor):
    # TODO speed up by merging all images, preprocessing them all together, and doing a row-wise ocr detection
    if callable(preprocessor) and preprocessor.__name__ == "<lambda>":
        frame_thresh = preprocessor(single_frame)
    else: # holds the fours parameters
        block_size, c_value, kernel_size, erosion_iterations = preprocessor[0], preprocessor[1], preprocessor[2], preprocessor[3]
        if not isinstance(block_size, int):
            block_size = np.round(block_size).astype(int)
        if not isinstance(kernel_size, int):
            kernel_size = np.round(kernel_size).astype(int)
        if not isinstance(erosion_iterations, int):
            erosion_iterations = np.round(erosion_iterations).astype(int)
        frame_thresh = apply_preprocessing(single_frame, block_size, c_value, kernel_size, erosion_iterations)
    ocr = ocr_func( cv2.cvtColor(frame_thresh, cv2.COLOR_GRAY2RGB) )
    return ocr, frame_thresh

def get_manual_ocr(image, frame_name, next_known, width=100, height=12):
    # Crop the image to remove all-white rows/columns
    rows, cols = np.any(image == 0, axis=1), np.any(image == 0, axis=0)
    cropped_image = image[np.ix_(rows, cols)]    

    # Rescale the cropped image to fixed size for command line output
    resized_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Display binary image as pixel art in the terminal
    print("-----------------------------------------------------")
    for y in range(height):
        print(''.join(list(map(lambda v: '█' if v == 0 else ' ', resized_image[y,:]))))

    # Prompt the user for manual OCR correction in the terminal
    corrected_text = input(f"\nPlease type the displayed float number and hit enter ({next_known}, current frame is {frame_name}): ")
    return corrected_text

def run_complete_ocr(preloaded, ocr_func, preprocessor, manual_correction=False, write_img=False):
    ocr_out, errors = {}, 0
    for idx, (frame_name, frame) in tqdm(enumerate(preloaded), total=len(preloaded), desc='Performing OCR across all images'):
        prev_name = frame_names[idx-1]
        fixed, val = False, np.nan
        ocr, prep_fr = detect_ocr(frame, ocr_func, preprocessor)
        try:
            assert len(ocr) == 5
            val = float(ocr)
            if idx > 0 and isinstance(ocr_out[prev_name]['value'], float):
                assert ocr_out[prev_name]['value'] <= val
        except Exception:
            errors += 1
            last_known = ocr_out[prev_name]['value'] if idx > 0 else 0
            if manual_correction:
                while not fixed:
                    ocr = get_manual_ocr(prep_fr, frame_name, f'last number was {last_known}')
                    try:
                        val = float(ocr)
                        fixed = True
                    except Exception:
                        print(f'Incorrect input "{ocr}"!')
            else:
                ocr, val, fixed = None, None, False
        ocr_out[frame_name] = {'ocr': ocr, 'value': val, 'manual': fixed}
        if write_img:
            ocr_fname = frame_name.replace('frame', 'ocr').replace('.jpg', f'_{ocr.replace(".", "-")}.jpg')
            cv2.imwrite(ocr_fname, prep_fr)
    return ocr_out, errors

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Match the report of an mlflow experiment with the captured webcam images.")
    argparser.add_argument("--interactive", default=False, help="Whether to run the interactive preprocessing parameter selection.")
    argparser.add_argument("--tesseract_path", default=None, type=str, help="Path to the tesseract executable (if not in PATH).")
    argparser.add_argument("--tesseract_custom_data", default=r"/home/fischer/repos/Tesseract_sevenSegmentsLetsGoDigital/Trained data", type=str, help="Path to the custom trained data for tesseract.")
    args = argparser.parse_args()

    for report_fname in os.listdir(os.path.join(os.path.dirname(__file__), 'results')):
        if 'csv' in report_fname and 'image_analysis' not in report_fname:
            if os.path.isfile(os.path.join(os.path.dirname(__file__), 'results', report_fname.replace('.csv', '_image_analysis.csv'))):
                continue

            # load report
            report = pd.read_csv(os.path.join(os.path.dirname(__file__), 'results', report_fname))
            report = report.dropna().set_index('run_id').sort_values('start_time')
            # make sure that paths align (could happen when analysis takes place on a separate machine)
            report['artifact_uri'] = report['artifact_uri'].apply(lambda x: os.path.join(os.path.dirname(__file__), 'mlruns', *x.split('mlruns/')[1].split('/')))
            img_dir = os.path.join(os.path.dirname(__file__), 'results', report_fname.replace('.csv', ''))
            if not os.path.isdir(img_dir):
                os.makedirs(img_dir, exist_ok=True)
            # load frame names
            frame_names = []
            for uri in report['artifact_uri']:
                for fname in ['capture_start.jpg', 'capture_stop.jpg']:
                    frame_names.append( os.path.join(img_dir, f'{os.path.basename(os.path.dirname(uri))}_{fname}') )
                    if not os.path.isfile(frame_names[-1]): # on execution environment: copy frames from mlflow logs
                        assert os.path.exists(os.path.join(uri, fname)), f'File {os.path.join(uri, fname)} does not exist!'
                        shutil.copyfile(os.path.join(uri, fname), frame_names[-1])
            
            # init tesseract
            if args.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
            try:
                pytesseract.image_to_string(cv2.imread(frame_names[0]))
            except Exception as e:
                print('There was an error with pytesseract - make sure to install tesseract and adjust the path correctly via --tesseract_path')
                print(e)
            os.environ['TESSDATA_PREFIX'] = args.tesseract_custom_data
            ocr_func = lambda im: pytesseract.image_to_string(im, lang='lets', config='--psm 6').replace('\n', '').replace(',', '.').replace(' ', '').replace('-', '')

            # use default roi and preprocessing, or finetune interactively
            x1, y1, x2, y2 = (260, 195, 401, 256)
            preprocessor = lambda im: apply_preprocessing(im, 21, 10, 1, 1)
            if args.interactive:
                roi = select_roi(cv2.imread(frame_names[0]))
                x1, y1, x2, y2 = roi
                preloaded = [(fname, cv2.imread(fname)[y1:y2, x1:x2]) for fname in frame_names]
                test_frames = [preloaded[idx][1] for idx in np.random.choice(np.arange(len(preloaded)), size=7)]
                while True:
                    # for idx, test_frame in enumerate(test_frames):
                    #     cv2.imshow(f"Test Frame {idx+1}", test_frame)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    preprocessor, params = interactive_preprocessing_with_ocr(test_frames, ocr_func)
                    ocr_out, errors = run_complete_ocr(preloaded, ocr_func, preprocessor)
                    print(f'ROI: {roi} PARAMS {params} REMAINING ERRORS {errors} ({errors/len(ocr_out)*100:3.2f}%)')
                    continue_yn = input(f"\nDo you want to change the processing parameters again (the remaining errors have to be corrected manually)? (y/n): ")
                    if continue_yn == 'n':
                        break
            else:
                preloaded = [(fname, cv2.imread(fname)[y1:y2, x1:x2]) for fname in frame_names]

            # run complete ocr detection with manual correction
            ocr_out, errors = run_complete_ocr(preloaded, ocr_func, preprocessor, manual_correction=True, write_img=True)

            # traverse backwards to find any new errors relating to manual correction
            print('Now traversing backwards to find additional errors')
            for idx, (frame_name, frame) in enumerate(reversed(preloaded)):
                if idx == len(ocr_out) - 1 or idx == 0:
                    continue
                prev_name, next_name = frame_names[len(ocr_out)-idx-2], frame_names[len(ocr_out)-idx]
                last, current, next = ocr_out[prev_name], ocr_out[frame_name], ocr_out[next_name]
                if current['value'] < last['value'] or current['value'] > next['value']:
                    error = False
                    ocr, prep_fr = detect_ocr(frame, ocr_func, preprocessor)
                    while not error:
                        manual_input = get_manual_ocr(prep_fr, frame_name, f'previous is {last["value"]}, next is {next["value"]}')
                        try:
                            ocr_out[frame_name]['value'] = float(manual_input)
                        except Exception:
                            print(f'Incorrect input "{manual_input}"!')
                        try:
                            assert ocr_out[frame_name]['value'] <= next['value']
                            ocr_out[frame_name]['manual'] = True
                            error = True
                            # delete and re-write already written ocr file
                            ocr_fname = frame_name.replace('frame', 'ocr').replace('.jpg', f'_{ocr.replace(".", "-")}.jpg')
                            os.remove(ocr_fname)
                            ocr_fname = frame_name.replace('frame', 'ocr').replace('.jpg', f'_{manual_input.replace(".", "-")}.jpg')
                            cv2.imwrite(ocr_fname, prep_fr)
                        except Exception:
                            print(f'Incorrect input - input number ({manual_input}) cannot be bigger than the following value ({next["value"]})!')

            # write the image analysis summary
            df = pd.DataFrame(ocr_out).transpose()
            df['val_diff'] = df["value"].diff()
            df['still_errors'] = df['val_diff'] < 0
            df.to_csv(os.path.join(os.path.dirname(__file__), 'results', report_fname.replace('.csv', '_image_analysis.csv')))
            if not df["value"].is_monotonic_increasing:
                print('Still encountered errors in the following rows and frames:\n')
                print(df[df['val_diff'] < 0].index)
