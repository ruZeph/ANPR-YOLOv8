import os
import OCR_Code
import pickle
import cv2
from imutils import paths
import anpr_YOLOv8 as ANPR
from essentials import mk_title
from essentials import show_img

if __name__ == '__main__':
    # This is for mass input
    IMG_PATH = 'bike_samples/'
    if not os.path.exists(IMG_PATH):
        print("Sample Images not found. Please check the path and try again.")
    file_paths = sorted(list(paths.list_images(IMG_PATH)), reverse=False)

    # Create main Results directory if it doesn't exist
    RESULTS_DIR = "Results/"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Get the license plates using Y0L0v8 Model in the Predefined Resolution
    for file_path in file_paths:
        # Path Preprocessing to save image
        path = str(file_path)
        remove = ".jpg"
        bare_name = path.replace(remove, '') # bare_name = Name OF Image without extension
        print(f"Processing: {path}")  # Fix the standalone print
        org_title = bare_name.replace(IMG_PATH, '')
        
        # Create individual results folders within the main Results directory
        op_path = os.path.join(RESULTS_DIR, org_title, "detection/")
        clean_img_path = os.path.join(RESULTS_DIR, org_title, "cleaned_plate/")    
        ocr_output_path = os.path.join(RESULTS_DIR, org_title, "ocr_output/")
        
        # Create directories if they don't exist
        for directory in [op_path, clean_img_path, ocr_output_path]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        org_img = cv2.imread(file_path)
        img_cvt = cv2.resize(org_img, (1280, 720))

        title = "0_" + "Original Image"
        # ANPR.show_img(title, img_cvt)
        output_txt = mk_title(org_title, title)
        ANPR.save_img(op_path, output_txt, img_cvt)

        # Run the YOLOv8 Model and Check if number plate is detected in input
        check = ANPR.get_license_plate(img_cvt)
        if not check:
            print("NO License plate was detected for: " + file_path)
            continue
        else:
            print("License plate detected for: " + file_path)

            # Assign the YOLOv8 Model detection results
            results, detected_box = check

            # Crop & Plot the License plate if detection was successful
            license_plate = ANPR.crop_plate(img_cvt, results, detected_box, op_path, org_title)

            # Resize the cropped plate for standardized size
            height, width, channels = license_plate.shape
            normal_format = cv2.resize(license_plate, (500, 250), interpolation=cv2.INTER_CUBIC)

            # Preprocess license plate
            preprocessed_image, grayscale = ANPR.preprocess_lPlate(normal_format, op_path, org_title)

            # Clean the license plate to remove White blobs(Connected Component analysis)
            clean_img = ANPR.clean_license_plate(preprocessed_image, grayscale.shape, clean_img_path, org_title)

            # Character Segmentation
            lPlate, segments, bounding_boxes = ANPR.segment_lic_plate(clean_img, op_path, org_title, clean_img_path)
            
            # Run OCR directly on the segmented image and bounding boxes
            result_img, license_number = OCR_Code.apply_ocr(lPlate, bounding_boxes, ocr_output_path, org_title)
            
            print("License Plate Number:", license_number)
            print("--------------------------------------------------")