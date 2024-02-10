import tensorflow as tf
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


from essentials import mk_title, show_img, save_img
import numpy as np

# Initialize tensorflow components
keras = tf.keras
load_model = keras.models.load_model
img_to_array = keras.preprocessing.image.img_to_array

count = 1



def apply_ocr(image, bounding_boxes, op_path=None, org_title=None):
    global count

    color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Define constants
    TARGET_WIDTH = 128
    TARGET_HEIGHT = 128

    chars = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    # Load the pre-trained convolutional neural network
    model = load_model('weights/characters_model.weights', compile=False)

    vehicle_plate_number = ""
    i = 1  # to keep track of number of characters
    # Loop over the bounding boxes
    for rect in bounding_boxes:
        # Get the coordinates from the bounding box
        x, y, w, h = rect

        # Crop the character from the mask
        # and apply bitwise_not because in our training data for pre-trained model
        # the characters are black on a white background
        crop = image[y:y + h, x:x + w]
        crop = cv2.bitwise_not(crop)

        # Get the number of rows and columns for each cropped image
        # and calculate the padding to match the image input of pre-trained model
        rows = crop.shape[0]
        columns = crop.shape[1]
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)

        # Apply padding to make the image fit for neural network model
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)

        # Convert and resize image
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

        crop_path = op_path + "/Cropped Characters/"+ org_title+"/"
        if not os.path.exists(crop_path):
            os.makedirs(crop_path)
        title = "Cropped Characters_" + str(i)
        show_img(title, crop)
        output_txt = mk_title(org_title, title)
        save_img(crop_path, output_txt, crop)
        i += 1

        # Prepare data for prediction
        crop = crop.astype("float") / 255.0
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)

        # Make prediction
        prob = model.predict(crop)[0]
        idx = np.argsort(prob)[-1]
        vehicle_plate_number += chars[idx]

        # Show bounding box and prediction on image
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_img, chars[idx], (x - 15, y), font, 0.8, (0, 0, 255), 2)

    title1 = org_title.replace('_Cleaned_License_Plate.jpg', '')
    title2 = "_Character Recognitions"
    count = 1
    show_img(title, color_img)
    output_txt = mk_title(title1, title2)
    save_img(op_path, output_txt, color_img)
    return color_img, vehicle_plate_number