import os

import cv2
from imutils import paths
import anpr_YOLOv8 as ANPR
from essentials import mk_title

if __name__ == '__main__':
    # This is for mass input
    IMG_PATH = 'bike_samples/'
    file_paths = sorted(list(paths.list_images(IMG_PATH)), reverse=False)
    # print(file_paths)


    # Get the license plates using Y0L0v8 Model in the Predefined Resolution
    for file_path in file_paths:
        # Path Preprocessing to save image
        path = str(file_path)
        remove = ".jpg"
        bare_name = path.replace(remove, '') # bare_name = Name OF Image without extension
        print
        op_path = "results/" + bare_name + '/'
        org_title = bare_name.replace(IMG_PATH, '')
        clean_img_path = "Cleaned_License_Plate/"

        if not os.path.exists(op_path):
            os.makedirs(op_path)
            
        if not os.path.exists(clean_img_path):
            os.makedirs(clean_img_path)

        org_img = cv2.imread(file_path)
        img_cvt = cv2.resize(org_img, (1280, 720))

        title = "0_" + "Original Image"
        # ANPR.show_img(title, img_cvt)/home/avisek
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

       

