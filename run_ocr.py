import OCR_Code
from imutils import paths
import cv2
import os
import pickle

if __name__ == '__main__':
    from essentials import show_img

    OCR_PATH = 'Cleaned_License_Plate/'
    BBOX_PATH = 'Cleaned_License_Plate/Bounding_Box/'
    file_paths = sorted(list(paths.list_images(OCR_PATH)), reverse=False)
    bbox_file_paths = sorted(([os.path.join(BBOX_PATH, filename) for filename in os.listdir(BBOX_PATH) if filename.endswith('.pkl')]), reverse=False)
    print(file_paths)
    print(bbox_file_paths)

    count = 1

    for file_path, bbox_file_path in zip(file_paths, bbox_file_paths):
        # Load the image
        print(file_path)
        clean_img = cv2.imread(file_path)
        show_img(str("Original Image" + str(count)),clean_img)
        count+=1    
        org_title = file_path.split('/')[-1]
        op_path = 'OCR_Output/'
        
        if not os.path.exists(op_path):
            os.makedirs(op_path)

        lPlate = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        
        with open(bbox_file_path, 'rb') as f:
            bounding_boxes = pickle.load(f)
        
        # Optical Character Recognition
        result_img, license_number = OCR_Code.apply_ocr(lPlate, bounding_boxes, op_path, org_title)
        OCR_Code.show_img("Result",result_img)

        print("License Plate Number: ", license_number)
        print("--------------------------------------------------")


