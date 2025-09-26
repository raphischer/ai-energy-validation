import argparse
import os
import shutil
import joblib

from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

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
    print("Draw a rectangle to select the region of interest (ROI) and hit space once satisfied.")
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

def apply_preprocessing(image, block_size, c_value, kernel_size, erosion_iterations, w_x1, w_xd, w_xs, w_y1, w_yd):
    # Ensure block_size is odd and greater than 1
    if block_size % 2 == 0:
        block_size += 1
    # resized = cv2.resize(image, (0, 0), fx=0.5, fy=0.5) # resizing makes everything faster -> TODO move outside of this function
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value) # threshold
    kernel = np.ones((kernel_size, kernel_size), np.uint8) # create erosion & dilation kernel
    dilated_image = cv2.dilate(thresh, kernel, iterations=erosion_iterations)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_iterations)
    digits, in_bounds, curr_x = [], True, w_x1
    while in_bounds:
        digits.append(eroded_image[w_y1:(w_y1+w_yd), curr_x:(curr_x+w_xd)])
        curr_x += w_xs
        if curr_x + w_xd >= eroded_image.shape[1]:
            in_bounds = False
    return digits, eroded_image

def update_preprocessing(x):
    try: # Get current positions of trackbars
        block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing')
        c_value = cv2.getTrackbarPos('C Value', 'Preprocessing')
        kernel_size = cv2.getTrackbarPos('Kernel Size', 'Preprocessing')
        erosion_iterations = cv2.getTrackbarPos('Erosion Iterations', 'Preprocessing')
        w_x1 = cv2.getTrackbarPos('Crop X Start', 'Preprocessing')
        w_xd = cv2.getTrackbarPos('Crop X Width', 'Preprocessing')
        w_xs = cv2.getTrackbarPos('Crop X Stride', 'Preprocessing')
        w_y1 = cv2.getTrackbarPos('Crop Y Start', 'Preprocessing')
        w_yd = cv2.getTrackbarPos('Crop Y Width', 'Preprocessing')
    except Exception:
        return # only happen during initialization of window

    digits = []
    for frame in frames:
        # Apply preprocessing with current parameters
        for digit, _ in apply_preprocessing(frame, block_size, c_value, kernel_size, erosion_iterations, w_x1, w_xd, w_xs, w_y1, w_yd):
            digits.append(cv2.cvtColor(digit, cv2.COLOR_GRAY2BGR))
    # Display the processed digits
    frames_arr = np.hstack(frames)
    # frames_arr = (frames_arr - np.min(frames_arr)) / (np.max(frames_arr) - np.min(frames_arr)) * 255
    digit_start_pos = np.linspace(0, frames_arr.shape[1]-digits[0].shape[1], len(digits))
    digits_arr = np.zeros(frames_arr.shape, dtype=frames_arr.dtype)
    for digit, x_start in zip(digits, digit_start_pos):
        digits_arr[0:digits[0].shape[0],int(x_start):int(x_start)+digits[0].shape[1]] = digit
    cv2.imshow("Preprocessing", np.vstack([frames_arr, digits_arr]))

def interactive_preprocessing(images):
    global frames
    frames = images.copy()  # Store the image globally for access inside trackbar callback

    # Create window and trackbars for adjusting preprocessing parameters
    cv2.namedWindow('Preprocessing', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Preprocessing', 1500, 800)
    cv2.createTrackbar('Block Size', 'Preprocessing', 21, 50, lambda x: update_preprocessing(x))
    cv2.createTrackbar('C Value', 'Preprocessing', 10, 20, lambda x: update_preprocessing(x))
    cv2.createTrackbar('Kernel Size', 'Preprocessing', 1, 20, lambda x: update_preprocessing(x))
    cv2.createTrackbar('Erosion Iterations', 'Preprocessing', 1, 10, lambda x: update_preprocessing(x))
    cv2.createTrackbar('Crop X Start', 'Preprocessing', 2, 20, lambda x: update_preprocessing(x))
    cv2.createTrackbar('Crop X Width', 'Preprocessing', 30, 50, lambda x: update_preprocessing(x))
    cv2.createTrackbar('Crop X Stride', 'Preprocessing', 36, 50, lambda x: update_preprocessing(x))
    cv2.createTrackbar('Crop Y Start', 'Preprocessing', 9, 20, lambda x: update_preprocessing(x))
    cv2.createTrackbar('Crop Y Width', 'Preprocessing', 42, 50, lambda x: update_preprocessing(x))
    update_preprocessing(0) # call once for initial display

    # Keep the window open until the user presses 'Esc'
    while True:
        if cv2.waitKey(1) & 0xFF == 32:  # Space key to exit
            break

    block_size = cv2.getTrackbarPos('Block Size', 'Preprocessing')
    c_value = cv2.getTrackbarPos('C Value', 'Preprocessing')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'Preprocessing')
    erosion_iterations = cv2.getTrackbarPos('Erosion Iterations', 'Preprocessing')
    w_x1 = cv2.getTrackbarPos('Crop X Start', 'Preprocessing')
    w_xd = cv2.getTrackbarPos('Crop X Width', 'Preprocessing')
    w_xs = cv2.getTrackbarPos('Crop X Stride', 'Preprocessing')
    w_y1 = cv2.getTrackbarPos('Crop Y Start', 'Preprocessing')
    w_yd = cv2.getTrackbarPos('Crop Y Width', 'Preprocessing')
    cv2.destroyAllWindows()
    prep_params = (block_size, c_value, kernel_size, erosion_iterations, w_x1, w_xd, w_xs, w_y1, w_yd)
    return lambda im: apply_preprocessing(im, block_size, c_value, kernel_size, erosion_iterations, w_x1, w_xd, w_xs, w_y1, w_yd), prep_params

def get_training_label(digits):
    n_expected = len(digits)
    digits = np.hstack(digits)
    digits = cv2.resize(digits, (100, int(60/digits.shape[1]*digits.shape[0])), interpolation=cv2.INTER_NEAREST)
    # Display binary image as pixel art in the terminal
    not_ok, last = True, None
    while not_ok:
        print("-----------------------------------------------------")
        for y in range(digits.shape[0]):
            print(''.join(list(map(lambda v: '█' if v == 0 else ' ', digits[y,:]))))
        # Prompt the user for manual OCR correction in the terminal
        last_s = f' - invalid input {last}' if last is not None else ''
        last = input(f"\nPlease type the {n_expected} displayed digits without spaces and hit enter{last_s}: ")
        try:
            assert len(last) == n_expected
            a = int(last)
            not_ok = False
        except Exception:
            pass
    return last

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

def run_complete_ocr(preloaded, ocr_func, manual_correction=False, write_img=False):
    ocr_out, errors = {}, 0
    for idx, (frame_name, frame) in tqdm(enumerate(preloaded), total=len(preloaded), desc='Performing OCR across all images'):
        prev_name = preloaded[idx-1][0]
        fixed, val = False, np.nan
        ocr, prep_fr = ocr_func(frame)
        try:
            assert len(ocr) == 5
            assert '.' in ocr
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

def sklearn_ocr(img, preprocessor, clf): # 2, 25, 35, 10, 40
    digits, prep_fr = preprocessor(img)
    images_np = np.array([i.flatten() for i in digits])
    pred_labels = clf.predict(images_np)
    return f'{pred_labels[0]}.{pred_labels[1]}{pred_labels[2]}{pred_labels[3]}', prep_fr

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Match the report of an mlflow experiment with the captured webcam images.")
    argparser.add_argument("--interactive", default=False, help="Whether to run the interactive preprocessing parameter selection.")
    argparser.add_argument("--ocr", default='results/ocr_random_forest.pkl', type=str, help="Path to an SKLEARN Classifier, or to the tesseract executable (if not in PATH).")
    args = argparser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))

    frame_names = {}
    # collect images for all results logs
    for report_fname in os.listdir(os.path.join(base_dir, 'results')):
        if 'csv' in report_fname and 'image_analysis' not in report_fname:
            if os.path.isfile(os.path.join(base_dir, 'results', report_fname.replace('.csv', '_image_analysis.csv'))):
                continue
            # load report
            report = pd.read_csv(os.path.join(base_dir, 'results', report_fname))
            report = report.dropna().set_index('run_id').sort_values('start_time')
            # make sure that paths align (could happen when analysis takes place on a separate machine)
            report['artifact_uri'] = report['artifact_uri'].apply(lambda x: os.path.join(base_dir, 'mlruns', *x.split('mlruns/')[1].split('/')))
            img_dir = os.path.join(base_dir, 'results', report_fname.replace('.csv', ''))
            os.makedirs(img_dir, exist_ok=True)
            # load frame names
            frame_names[report_fname] = []
            for uri in report['artifact_uri']:
                for fname in ['capture_start.jpg', 'capture_stop.jpg']:
                    full_fname = os.path.join(img_dir, f'{os.path.basename(os.path.dirname(uri))}_{fname}')
                    frame_names[report_fname].append(full_fname)
                    if not os.path.isfile(full_fname): # on execution environment, this will copy frames from mlflow logs
                        assert os.path.exists(os.path.join(uri, fname)), f'File {os.path.join(uri, fname)} does not exist!'
                        shutil.copyfile(os.path.join(uri, fname), full_fname)
    all_fnames = [fname for e_fnames in frame_names.values() for fname in e_fnames]

    # use default roi and preprocessing, or finetune interactively
    x1, y1, x2, y2 = (260, 195, 401, 256)
    preprocessor = lambda im: apply_preprocessing(im, 21, 10, 1, 1, 2, 30, 36, 9, 42)
    # params: block_size, c_value, kernel_size, erosion_iterations, w_x1, w_xd, w_xs, w_y1, w_yd
    if args.interactive:
        roi = select_roi(cv2.imread(np.random.choice(all_fnames, size=1)[0]))
        x1, y1, x2, y2 = roi
        preloaded = [(fname, cv2.imread(fname)[y1:y2, x1:x2]) for fname in all_fnames]
        test_frames = [preloaded[idx][1] for idx in np.random.choice(np.arange(len(preloaded)), size=7)]
        preprocessor, prep_params = interactive_preprocessing(test_frames)

    # load ocr model or train a new one
    clf, ocr_func = None, None
    try:
        with open(args.ocr, "rb") as f:
            clf = joblib.load(f)
        ocr_func = lambda im: sklearn_ocr(im, preprocessor, clf)
        print(f'Loaded pre-trained OCR random forest from {args.ocr}')
    except Exception:
        print(f'Could not load OCR model {args.ocr} - starting interactive labeling to create a new random forest model!')
        # initilization and hyperparameters
        continue_training, all_digits, X_data, y_data = True, [], [], []
        for fname in all_fnames:
            all_digits.extend(preprocessor(cv2.imread(fname)[y1:y2, x1:x2])[0])
        while continue_training:
            # get new labels
            to_label = [all_digits[idx] for idx in np.random.choice(np.arange(len(all_digits)), size=8)]
            X_data.extend(to_label)
            labels = get_training_label(to_label)
            y_data.extend(labels)
            # train model and check for cross-validated accuracy
            samples, labels = np.array([img.flatten() for img in X_data]), np.array(y_data)
            labels_enc = LabelEncoder().fit_transform(labels)
            try:
                acc_score = cross_val_score(RandomForestClassifier(), samples, labels_enc)
                # check if result is sufficient
                cont = input(f'With {len(labels)} labels ({len(labels)/len(all_digits)*100:3.2f}% of the available data), a cross-validated accuracy of {np.mean(acc_score)*100:.3f}% could be achieved. Hit enter to continue labeling, or input "stop" to stop labeling: ')
                continue_training = cont.lower().strip()!='stop'
            except ValueError:
                pass
        # train final classifier
        final_rf = RandomForestClassifier()
        final_rf.fit(samples, labels_enc)
        joblib.dump(final_rf, os.path.join('results', 'final_random_forest.pkl'))
        ocr_func = lambda im: sklearn_ocr(im, preprocessor, final_rf)

    # process each individual run
    for report_fname, frames in frame_names.items():

        # load and run complete ocr detection with manual correction
        preloaded = [(fname, cv2.imread(fname)[y1:y2, x1:x2]) for fname in frames]
        ocr_out, errors = run_complete_ocr(preloaded, ocr_func, manual_correction=False, write_img=False)
        print(f'Found {errors} errors ({errors/len(ocr_out)*100:3.2f}%) that need to be manually fixed')
        ocr_out, errors = run_complete_ocr(preloaded, ocr_func, manual_correction=True, write_img=True)

        # traverse backwards to find any new errors relating to manual correction
        print('Now traversing backwards to find additional errors')
        for idx, (frame_name, frame) in enumerate(reversed(preloaded)):
            if idx == len(ocr_out) - 1 or idx == 0:
                continue
            prev_name, next_name = preloaded[len(ocr_out)-idx-2][0], preloaded[len(ocr_out)-idx][0]
            last, current, next = ocr_out[prev_name], ocr_out[frame_name], ocr_out[next_name]
            if current['value'] < last['value'] or current['value'] > next['value']:
                error = False
                ocr, prep_fr = ocr_func(frame)
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
        df.to_csv(os.path.join(base_dir, 'results', report_fname.replace('.csv', '_image_analysis.csv')))
        if not df["value"].is_monotonic_increasing:
            print('Still encountered errors in the following rows and frames:\n')
            print(df[df['val_diff'] < 0].index)
