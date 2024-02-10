This is my Bachelors degree project. It implements YOLOv8 and a CNN based custom OCR model to perform Automatic Number Plate Detection(ANPR) on Indian four wheelers.

The system is optimized to perform OCR on 2-Row Indian HSRNs(High Security Number Plate).

The model detects 1-Row HSRNs(from Bikes) but to doesn't perform OCR correctly.

It doesn't detect Indian Green HSRNs(for EVS) at all.

---

### Some things to Note

1. The license detection model is completely trained by me using YOLOv8 in Google Collab. You can check the `YOLO_Models` folder for the code.
2. The OCR Model & Implementation I used is from the post [Characters Segmentation and Recognition for Vehicle License Plate](http://dangminhthang.com/knowledge-sharing/characters-segmentation-and-recognition-for-vehicle-license-plate/)  by Minh Thang Dang(Thanks !!!).

---

### Important !!! Read it

When I implemented the project, I used Tensorflow for GPU v2.12 and PyTorch on CUDA 11.8, which did not conflict at all. But if you try to install current  `tensorflow[and-cuda]` package , they will conflict. So you have to set up two different environments to run the project , i.e., one for the License Plate Detection and  one for OCR.

---

### How to implement the code?

**USE PYTHON 3.11**

<u>STEP - 01 : FOR DETECTION</u>

1. Clone the repo (Beware.. It's nearly ~ 4 GB )
2. Make a python virtual environment named `Detection`. Copy following files & folder into it:
 1. bike_samples/
 2. weights/
 3. anpr_YOLOv8.py
 4. essentials.py
 5. main_detection.py
3. Install following python packages:
 1. PyTorch for CUDA 12.1
 2. imutils
 3. ultralytics
 4. deskew
4. Run the main_detection.py. You can see the output in `results/` folder.

<u>STEP - 02 : FOR OCR</u>

1. Make another virtual environment named `OCR`.
2. From the above run, you will get a folder named  `Cleaned_License_Plate`.  Copy that to OCR/ directory.
3. Now from the cloned repo, copy following files & folders into OCR/:
 1. weights/
 2. essentials.py
 3. OCR_Code.py
 4. run_ocr.py
4. Download the character_model weight from [here](https://mega.nz/file/6ZclwBoC#CoSjE68a6P85UHDIYhPL26IjCyssvy7pL6vwxs-xGKw). Copy it to weights/ directory.
5. Run the code `run_ocr.py`.

---

I have also implemented a code for detection in Videos. You can the code check the folder `Video_Detection`. You can download samples from [here](https://mega.nz/folder/yYNATSab#_joN65RZaTYx8PvoovLZlQ).
