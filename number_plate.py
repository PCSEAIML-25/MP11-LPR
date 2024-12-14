import cv2
import pytesseract
import re
import numpy as np

# Path to the Haar Cascade for detecting plates
haarcascade = "model/haarcascade_russian_plate_number.xml"

# Path to Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def validate_plate(plate_text):
    # Replace common OCR mistakes
    plate_text = plate_text.replace('O', '0').replace('I', '1').replace('Z', '2')

    # Relaxed format validation
    match = re.match(r'^[A-Z0-9]{6,12}$', plate_text)  # Adjust length and characters as needed

    if match:
        return plate_text
    else:
        return None

def process_image(image):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade classifier for plate detection
    plate_cascade = cv2.CascadeClassifier(haarcascade)

    # Detect plates in the grayscale image
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    plate_numbers = []

    min_area = 500  # Minimum area to consider as a plate

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Crop the detected plate from the image
            img_roi = image[y:y + h, x:x + w]

            # Preprocess the cropped image for OCR
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
            img_roi_blur = cv2.GaussianBlur(img_roi_gray, (5, 5), 0)
            img_roi_thresh = cv2.adaptiveThreshold(img_roi_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Clean image with morphological transformations
            kernel = np.ones((3, 3), np.uint8)
            img_roi_clean = cv2.morphologyEx(img_roi_thresh, cv2.MORPH_CLOSE, kernel)

            # Save the preprocessed image for debugging
            cv2.imwrite('debug_plate_clean.jpg', img_roi_clean)

            # Use Tesseract OCR with a whitelist of characters (only alphanumeric)
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            plate_text = pytesseract.image_to_string(img_roi_clean, config=custom_config)

            # Debug: Log extracted text
            print(f"Extracted Text: {plate_text.strip()}")

            # Filter out only alphanumeric characters using regex
            alphanumeric_texts = re.findall(r'\b[A-Z0-9]+\b', plate_text)

            if alphanumeric_texts:
                # Find the longest text which is likely the plate number
                most_relevant_text = max(alphanumeric_texts, key=len)

                # Validate the plate text
                valid_plate = validate_plate(most_relevant_text)
                if valid_plate:
                    plate_numbers.append(valid_plate)

    return plate_numbers
