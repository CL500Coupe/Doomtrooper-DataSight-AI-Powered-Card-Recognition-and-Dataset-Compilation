import pytesseract
import cv2
import numpy as np
import os
from PIL import Image
import re
import pandas as pd

# Path to Tesseract executable (same as in your code)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Dictionary of attribute template image paths
attribute_template_paths = {
    "walka_wrecz.png": r'D:\doomtrooper\to_find\atrybuty\walka_wrecz.png',
    "strzelanie.png": r'D:\doomtrooper\to_find\atrybuty\strzelanie.png',
    "pancerz.png": r'D:\doomtrooper\to_find\atrybuty\pancerz.png',
    "wartosc.png": r'D:\doomtrooper\to_find\atrybuty\wartosc.png',
}


# Digit template folder
digit_template_folder = r'D:\doomtrooper\to_find\liczby'

# Example usage
image_path = r'D:\doomtrooper\cards\dt_11.jpg'




found_path = r'D:\doomtrooper\found'


fraction_template_folder = r'D:\doomtrooper\to_find\frakcje'

feature_template_folder = r'D:\doomtrooper\to_find\cechy'

# Base directory for cards
cards_folder = r'D:\doomtrooper\cards'


def crop_card_from_image(image_path, attribute_template_paths, digit_template_folder, found_path):
    try:
        # Load the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Assuming the largest contour is the card
        card_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(card_contour)

        # Crop the card image
        cropped_card = img[y:y+h, x:x+w].copy()

        # Define and calculate the positions and sizes of the name and text areas
        top_percentage, height_percentage, width_percentage, left_percentage = 0.02, 0.15, 0.42, 0.25
        text_bottom_percentage, text_height_percentage = 0.07, 0.25

        name_top, name_height, name_width, name_left = int(h * top_percentage), int(h * height_percentage), int(w * width_percentage), int(w * left_percentage)
        text_top, text_height = h - int(h * text_bottom_percentage) - int(h * text_height_percentage), int(h * text_height_percentage)

        name_area = cropped_card[name_top:name_top + name_height, name_left:name_left + name_width]
        text_area = cropped_card[text_top:text_top + text_height, 0:w].copy()

        # Check for attributes in the text area
        attribute_found = False
        for attribute_name in attribute_template_paths.keys():
            template_path = attribute_template_paths[attribute_name]
            template = cv2.imread(template_path)
            result = cv2.matchTemplate(text_area, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            threshold = 0.6
            if max_val >= threshold:
                attribute_found = True
                break

        # If an attribute is found, remove the left portion of the text area
        if attribute_found:
            remove_width = int(w * 0.25)  # 15% of the card's width
            text_area[:, 0:remove_width] = 0  # Set the left portion to black

        # Generate file names using found_path
        base_filename = os.path.basename(image_path).replace('.jpg', '')
        cropped_card_path = os.path.join(found_path, f'{base_filename}_cropped.jpg')
        cropped_name_path = os.path.join(found_path, f'{base_filename}_name.jpg')
        cropped_text_path = os.path.join(found_path, f'{base_filename}_text.jpg')

        # Save the cropped card, name area, and modified text area images
        cv2.imwrite(cropped_card_path, cropped_card)
        cv2.imwrite(cropped_name_path, name_area)
        cv2.imwrite(cropped_text_path, text_area)

        return cropped_card_path, cropped_name_path, cropped_text_path
    except Exception as e:
        return f"Error: {e}", None, None

# Usage
cropped_card_path, cropped_name_path, cropped_text_path = crop_card_from_image(image_path, attribute_template_paths, digit_template_folder, found_path)
print("Cropped Card Image:", cropped_card_path)
print("Cropped Name Area Image:", cropped_name_path)
print("Cropped Text Area Image:", cropped_text_path)


def preprocess_image(image_path, save_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, processed_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(save_path, processed_img)
    except Exception as e:
        return f"Error: {e}"



# Function to read text from an image (modified to specify the type of image)
def read_text_from_image(image_base_path, found_path, image_type='_text.jpg'):
    base_filename = os.path.basename(image_base_path).replace('.jpg', '')
    text_image_path = os.path.join(found_path, f'{base_filename}{image_type}')
    try:
        img = Image.open(text_image_path)
        text = pytesseract.image_to_string(img, lang='pol')

        # Replace newlines and multiple spaces with a single space
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    except Exception as e:
        return f"Error: {e}"


# Function to read the card name from the cropped name area image
def find_card_name(image_base_path, found_path):
    base_filename = os.path.basename(image_base_path).replace('.jpg', '')
    name_image_path = os.path.join(found_path, f'{base_filename}_name.jpg')
    try:
        img = Image.open(name_image_path)
        card_name = pytesseract.image_to_string(img, lang='pol')

        # Replace newlines and multiple spaces with a single space
        card_name = re.sub(r'\n+', ' ', card_name)
        card_name = re.sub(r'\s+', ' ', card_name).strip()

        return card_name
    except Exception as e:
        return f"Error: {e}"



# Function to find any of the specified fractions in the image
def find_fraction(image_path, fraction_template_folder):
    try:
        img = cv2.imread(image_path)
        
        # Iterate through all fraction templates in the specified folder
        for fraction_filename in os.listdir(fraction_template_folder):
            template_path = os.path.join(fraction_template_folder, fraction_filename)
            template = cv2.imread(template_path)
            
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            threshold = 0.8  # You may adjust this threshold as needed
            if max_val >= threshold:
                return f"{fraction_filename.replace('.png', '')}"
        
        return "No matching fraction found"
    except Exception as e:
        return f"Error: {e}"
    
def find_feature(image_path, feature_template_folder):
    try:
        img = cv2.imread(image_path)
        
        # Iterate through all feature templates in the specified folder
        for feature_filename in os.listdir(feature_template_folder):
            template_path = os.path.join(feature_template_folder, feature_filename)
            template = cv2.imread(template_path)
            
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            threshold = 0.8  # Adjust this threshold as needed
            if max_val >= threshold:
                return f"{feature_filename.replace('.png', '')}"
        
        return "No matching feature found"
    except Exception as e:
        return f"Error: {e}"



def read_digit_from_cropped_image(attribute_name, digit_template_folder, save_debug_images=True):
    try:
        # Corrected path for the preprocessed image
        preprocessed_cropped_image_path = os.path.join(r'D:\doomtrooper\found', f"{attribute_name}_preprocessed.png")

        # Load the preprocessed cropped image in grayscale
        img = cv2.imread(preprocessed_cropped_image_path, cv2.IMREAD_GRAYSCALE)

        max_match_score = 0
        matched_digit = None

        # Iterate through digit templates
        for digit in range(1, 8):
            template_path = os.path.join(digit_template_folder, f"{digit}.png")
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > max_match_score:
                max_match_score = max_val
                matched_digit = digit

            # Save debug images if required
            if save_debug_images:
                debug_image_path = os.path.join(r'D:\doomtrooper\found\debug', f"{attribute_name}_{digit}_match.png")
                cv2.imwrite(debug_image_path, result)

        return matched_digit if matched_digit is not None else "Digit not found"
    except Exception as e:
        return f"Error: {e}"


# Modified function to find specific attributes and crop the region to the right of the attribute
def find_attributes(image_path, attribute_template_paths, digit_template_folder):
    attribute_readable_names = {
        "walka_wrecz.png": "WALKA WRECZ",
        "strzelanie.png": "STRZELANIE",
        "pancerz.png": "PANCERZ",
        "wartosc.png": "WARTOSC"
    }

    attributes = {}
    try:
        img = cv2.imread(image_path)
        for attribute_filename, readable_name in attribute_readable_names.items():
            template_path = attribute_template_paths.get(attribute_filename, None)
            
            if template_path is not None:
                template = cv2.imread(template_path)
                result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                threshold = 0.6
                if max_val >= threshold:
                    #print(f"{readable_name} found")
                    
                    h, w, _ = template.shape
                    x, y = max_loc
                    cropped = img[y:y+h, x+w:x+w+170]  # Adjust the width as needed
                    # Adjusted file naming for cropped image
                    cropped_filename = f"{attribute_filename.replace('.png', '')}_crop.png"
                    cropped_path = os.path.join(r'D:\doomtrooper\found', cropped_filename)
                    cv2.imwrite(cropped_path, cropped)
                    
                    # Preprocess and save the cropped image
                    preprocessed_filename = f"{attribute_filename.replace('.png', '')}_preprocessed.png"
                    preprocessed_path = os.path.join(r'D:\doomtrooper\found', preprocessed_filename)
                    preprocess_image(cropped_path, preprocessed_path)

                    # Read digit from the preprocessed image
                    digit = read_digit_from_cropped_image(attribute_filename.replace('.png', ''), digit_template_folder)
                    attributes[readable_name] = digit
                else:
                    print(f"{readable_name} not found")
    except Exception as e:
        print(f"Error: {e}")
    
    return attributes


def write_to_csv_with_pandas(data, csv_file_path):
    try:
        # Convert the data dictionary to a DataFrame
        new_data_df = pd.DataFrame([data])

        # Check if the CSV file already exists
        if os.path.isfile(csv_file_path):
            # Load existing data
            df = pd.read_csv(csv_file_path)
            # Append new data
            df = pd.concat([df, new_data_df], ignore_index=True)
        else:
            # Use new data as the DataFrame if the file doesn't exist
            df = new_data_df

        # Write to CSV
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
    except Exception as e:
        print(f"Error using Pandas to write to CSV: {e}")



# Iterate through each card
#for i in range(1, 15):  # Assuming card names go from dt_1.jpg to dt_14.jpg
#    image_path = os.path.join(cards_folder, f'dt_{i}.jpg')
#    print(f"Processing {image_path}")

    # Run your existing processing pipeline on each card
#    cropped_card_path, cropped_name_path, cropped_text_path = crop_card_from_image(image_path, attribute_template_paths, digit_template_folder, found_path)
#    card_name = find_card_name(image_path, found_path)
#    extracted_text = read_text_from_image(image_path, found_path)
#    fraction_match = find_fraction(cropped_card_path, fraction_template_folder)
#    feature_match = find_feature(cropped_card_path, feature_template_folder)
#    attributes = find_attributes(cropped_card_path, attribute_template_paths, digit_template_folder)

    # Prepare data for CSV
#    data = {
#        'name': card_name,
#        'text': extracted_text,
#        'fraction': fraction_match,
#        'feature': feature_match,
#        'walka_wrecz': attributes.get('WALKA WRECZ', '0'),
#        'strzelanie': attributes.get('STRZELANIE', '0'),
#        'pancerz': attributes.get('PANCERZ', '0'),
#        'wartosc': attributes.get('WARTOSC', '0')
#    }

    # Write data to CSV
#    csv_file_path = r'D:\doomtrooper\data.csv'
#    write_to_csv_with_pandas(data, csv_file_path)
#
#    print(f"Finished processing {image_path}\n")



# Using the function to read the card name
card_name = find_card_name(image_path, found_path)
print("\n\nCard Name:", card_name,'\n')


# Using the function to read text from the text area of the card
extracted_text = read_text_from_image(image_path, found_path)
print(f"Extracted Text: {extracted_text}\n")

# Find fractions and attributes using the cropped card image
fraction_match = find_fraction(cropped_card_path, fraction_template_folder)
print(fraction_match,'\n')

feature_match = find_feature(cropped_card_path, feature_template_folder)
print(feature_match,'\n')


attributes = find_attributes(cropped_card_path, attribute_template_paths, digit_template_folder)

for attribute, digit in attributes.items():
    print(f"{attribute}: {digit}")


data = {
    'name': card_name,
    'text': extracted_text,
    'fraction': fraction_match,
    'feature': feature_match,
    'walka_wrecz': attributes.get('WALKA WRECZ', '0'),
    'strzelanie': attributes.get('STRZELANIE', '0'),
    'pancerz': attributes.get('PANCERZ', '0'),
    'wartosc': attributes.get('WARTOSC', '0')
}

csv_file_path = r'D:\doomtrooper\data.csv'
write_to_csv_with_pandas(data, csv_file_path)


