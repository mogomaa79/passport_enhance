import os
import subprocess
from PIL import Image, ImageEnhance, ImageStat
import numpy as np
import cv2
from retinaface import RetinaFace
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from googleapiclient.http import MediaFileUpload


def checkMOHRE(passportNumber, 
               passport_file_path='passport.jpg', 
               photo_file_path='photo.jpg',
               nationality=['237', 'PHILIPPINES'], 
            #    nationality=['317', 'ATHYUOBYA'], 
               email='thepunicher86@gmail.com', 
               phoneNumber='0581231234'):
    # Build the curl command dynamically
    curl_command = [
        "curl", "--location",
        "https://eservices.mohre.gov.ae/TasheelWeb/services/transactionentry/505",
        "--header", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0",
        "--header", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "--header", "Accept-Language: en-US,en;q=0.5",
        "--header", "Accept-Encoding: gzip, deflate, br, zstd",
        "--header", "Origin: https://eservices.mohre.gov.ae",
        "--header", "Connection: keep-alive",
        "--header", "Referer: https://eservices.mohre.gov.ae/TasheelWeb/services/transactionentry/505",
        "--header", "Cookie: X-Language=en; __RequestVerificationToken_L1Rhc2hlZWxXZWI1=_jB0aJiVPIa2Dy3eOTHitukWRnLihlLGR2nRaBnXVNt0eI20PoM-ADMoIfgydEu_Cu1kXsUVPoWSVm9Q0gpq7j4R4ec1; JSS=02531ebdde-a4a8-48TmYCrmCEkJ-HkBM_GXg64-iIaS6C3h0w9IBzWRK53iBq86Vc-UCQmqa4PET-e86SziY",
        "--header", "Upgrade-Insecure-Requests: 1",
        "--header", "Sec-Fetch-Dest: document",
        "--header", "Sec-Fetch-Mode: navigate",
        "--header", "Sec-Fetch-Site: same-origin",
        "--header", "Sec-Fetch-User: ?1",
        "--header", "Priority: u=0, i",
        "--form", f'__RequestVerificationToken="JkEaRJKKY5lxJqeHjpAHSKGxSxjWO10hVHmTIbTdcs8pXS2z8I8JEtlwsKgXZ5Pc1vkRfRYG389ENfrUYxiAI5BqWDU1"',
        "--form", f'AttachmentType="PassportAndPersonPhoto"',
        "--form", f'PassportNumber="{passportNumber}"',
        "--form", f'Nationality.Value="{nationality[0]}"',
        "--form", f'Nationality.Description="{nationality[1]}"',
        "--form", f'TravelNationality.Value=""',
        "--form", f'TravelNationality.Description=""',
        "--form", f'Email="{email}"',
        "--form", f'ContactNo="{phoneNumber}"',
        "--form", f'EducationCertificateAvailable="false"',
        "--form", f'submitButton="Submit"',
        "--form", f'PassportDocumentFirstPage=@{passport_file_path}',
        "--form", f'PersonPhotoDocument=@{photo_file_path}'
    ]

    try:
        # Execute the curl command
        result = subprocess.run(curl_command, text=True, capture_output=True, check=True)
        return True
        # # Check for the presence of the string
        # if result.returncode == 0:  # A successful subprocess call typically has returncode 0
        #     # Optionally, you can parse the response for HTTP status code (if included)
        #     if "200 OK" in result.stdout or "HTTP/1.1 200" in result.stdout:
        #         return True
        # return False
    except subprocess.CalledProcessError as e:
        print("Error executing curl command:")
        return False

def upscale(image_path, output_path="passport_ph.jpg", scale_percent = 125):
    if os.path.getsize(image_path) < 200 * 1024:
        image = cv2.imread(image_path)
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # Resize the image
        upscaled_image = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(output_path, upscaled_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return
    image = cv2.imread(image_path)
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    

def enhance_contrast(image_path, output_path="passport_ph.jpg", contrast_factor=1.5):
    pil_image = Image.open(image_path)
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_pil_image = enhancer.enhance(contrast_factor)
    enhanced_pil_image.save(output_path, quality=100, subsampling=0)

def enhance_brightness(image_path, output_path="passport_ph.jpg", target_brightness=170):
    pil_image = Image.open(image_path)

    # Measure the current brightness
    stat = ImageStat.Stat(pil_image.convert("L"))
    current_brightness = stat.mean[0]  # Mean brightness value

    # Check if the brightness is within the desired range (150 to 190)
    if 150 <= current_brightness <= 190:
        # Save the image as it is
        pil_image.save(output_path, quality=100, subsampling=0)
        print(f"Brightness is already in the range ({current_brightness}). Image saved without modification.")
    else:
        # Adjust the brightness
        brightness_factor = target_brightness / current_brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        adjusted_image = enhancer.enhance(brightness_factor)

        # Save the adjusted image
        adjusted_image.save(output_path, quality=100, subsampling=0)
        print(f"Brightness adjusted from {current_brightness} to {target_brightness}. Image saved.")

def enhance_white_balance(image_path, output_path="passport_ph.jpg"):
    """
    Enhances the white balance of an image using OpenCV.
    
    :param image_path: Path to the input image.
    :param output_path: Path to save the white-balanced image.
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into L, A, B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge the LAB channels
    lab = cv2.merge((l, a, b))
    
    # Convert back to BGR color space
    balanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Save the white-balanced image
    cv2.imwrite(output_path, balanced_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"White-balanced image saved to {output_path}")

def enhance_sharpness(image_path, output_path="passport_ph.jpg", sharpness_factor=2.0):
    pil_image = Image.open(image_path)
    enhancer = ImageEnhance.Sharpness(pil_image)
    enhanced_pil_image = enhancer.enhance(sharpness_factor)
    enhanced_pil_image.save(output_path, quality=100, subsampling=0)

def correct_orientation(image_path, output_path="passport_photo.jpg"):
    # Load the image
    original_image = Image.open(image_path)
    image_np = np.array(original_image)

    # Detect faces and landmarks
    detections = RetinaFace.detect_faces(image_path)
    output_image = original_image

    if not detections:
        print("No faces detected!")

    else:
        face_key, face_data = next(iter(detections.items()))
        # Extract landmarks
        landmarks = face_data["landmarks"]
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        mouth_left = landmarks["mouth_left"]
        mouth_right = landmarks["mouth_right"]
        eye_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )
        mouth_center = (
            (mouth_left[0] + mouth_right[0]) / 2,
            (mouth_left[1] + mouth_right[1]) / 2,
        )
        eyes_distancex = abs(right_eye[0] - left_eye[0])
        eye_mouth_distancex = abs(mouth_center[0] - eye_center[0])

        # Determine orientation
        if eyes_distancex > eye_mouth_distancex and mouth_center[1] > eye_center[1]:
            print("Face is correctly oriented.")
            output_image =  original_image
        elif eyes_distancex < eye_mouth_distancex and mouth_center[0] < eye_center[0]:
            print("Face is right. Rotating 90°...")
            corrected_image = original_image.rotate(90, expand=True)
            output_image =  corrected_image
        elif eyes_distancex > eye_mouth_distancex and mouth_center[1] < eye_center[1]:
            print("Face is upside down. Rotating 180°...")
            corrected_image = original_image.rotate(-180, expand=True)
            output_image =  corrected_image
        elif eyes_distancex < eye_mouth_distancex and mouth_center[0] > eye_center[0]:
            print("Face is left. Rotating -90°...")
            corrected_image = original_image.rotate(-90, expand=True)
            output_image =  corrected_image
        else:
            print("Couldn't check face orientation...")
            output_image =  original_image
    output_image.save(output_path, quality=100, subsampling=0)
            
def crop_based_on_eyes(image_path, output_path="passport_photo.jpg"):
    # Load the image
    original_image = Image.open(image_path)
    image_np = np.array(original_image)

    # Detect faces and landmarks
    detections = RetinaFace.detect_faces(image_path)
    output_image = original_image

    if not detections:
        print("No faces detected!")

    else:
        selected_face = None
        min_left_eye_x = float('inf')

        for face_key, face_data in detections.items():
            left_eye_x = face_data["landmarks"]["left_eye"][0]
            if left_eye_x < min_left_eye_x:
                min_left_eye_x = left_eye_x
                selected_face = face_data
        # Extract landmarks
        landmarks = selected_face["landmarks"]
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]

        # Calculate the distance between the eyes
        eye_distance = np.sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)

        # Define cropping dimensions
        left = left_eye[0] - 4 * eye_distance
        right = right_eye[0] + 13 * eye_distance
        top = min(left_eye[1], right_eye[1]) - 4 * eye_distance
        bottom = max(left_eye[1], right_eye[1]) + 7 * eye_distance

        # Ensure crop box stays within image bounds
        width, height = original_image.size
        left = max(0, left)
        right = min(width, right)
        top = max(0, top)
        bottom = min(height, bottom)

        # Crop the image
        cropped_image = original_image.crop((left, top, right, bottom))
        print(f"Cropping region: left={left}, top={top}, right={right}, bottom={bottom}")

        # Save the cropped image
        output_image = cropped_image
    output_image.save(output_path, quality=100, subsampling=0)

def compress_image_set_dpi_save(image_path, output_path="passport_photo.jpg"):
    image_pil = Image.open(image_path)
    # Initialize variables
    target_size_kb = 900
    max_quality=100
    min_quality=10
    quality = max_quality
    step = 5

    while quality >= min_quality:

        # Save image with the current quality setting
        temp_path = output_path
        image_pil.save(temp_path, format="JPEG", quality=quality, dpi=(600, 600))

        # Check the file size
        file_size_kb = os.path.getsize(temp_path) / 1024  # Convert to KB
        print(f"file size is {file_size_kb}kb")
        if file_size_kb <= target_size_kb:
            print(f"Compressed successfully to {file_size_kb:.2f} KB with quality = {quality}")
            return temp_path

        # Decrease quality for the next iteration
        quality -= step

    print("Could not compress the image to the target size with sufficient quality.")
    return temp_path

def upload_to_drive(file_path, folder_id, service):
    """
    Uploads a file to Google Drive and returns the public URL.

    :param file_path: Path to the local file.
    :param folder_id: Google Drive folder ID to upload the file to.
    :param service: Authenticated Google Drive service instance.
    :return: Public URL of the uploaded file.
    """
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    # Make the file public
    file_id = uploaded_file.get('id')
    service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()

    # Return the public URL
    return f"https://drive.google.com/uc?id={file_id}"

def upload_to_drive(file_path, folder_id, service):
    """
    Uploads a file to Google Drive and returns the public URL.

    :param file_path: Path to the local file.
    :param folder_id: Google Drive folder ID to upload the file to.
    :param service: Authenticated Google Drive service instance.
    :return: Public URL of the uploaded file.
    """
    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    # Make the file public
    file_id = uploaded_file.get('id')
    service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()

    # Return the public URL
    return f"https://drive.google.com/uc?id={file_id}"

def process_image_combinations(folder, sheet_id="1et9BQ8NtRMBsLauEr0n22ptitruoBPkOMCVFB6a6n0k",
                               service_account_file="service_account_key.json",
                               drive_folder_id="1eErpfx3-sZoDvC5v9n4LWC-3NQfIlrLa"):
    """
    Processes image combinations and logs results to a Google Sheet with embedded images.

    :param folder: Folder containing the input images.
    :param sheet_id: ID of the Google Sheet to log results.
    :param service_account_file: Path to the service account JSON file for Google Sheets API.
    :param drive_folder_id: Google Drive folder ID to upload images to.
    """
    # Authenticate with Google APIs
    creds = Credentials.from_service_account_file(service_account_file)
    sheets_service = build('sheets', 'v4', credentials=creds)
    drive_service = build('drive', 'v3', credentials=creds)
    sheet = sheets_service.spreadsheets()

    # File paths
    passport_path = os.path.join(folder, 'passport.jpg')
    cropped_path = os.path.join(folder, 'crop.jpg')
    upscale_path = os.path.join(folder, 'upscale.jpg')
    white_balance_path = os.path.join(folder, 'white_balance.jpg')
    brightness_path = os.path.join(folder, 'brightness.jpg')
    contrast_path = os.path.join(folder, 'contrast.jpg')
    sharpened_path = os.path.join(folder, 'sharpened.jpg')
    dpi_path = os.path.join(folder, 'dpi.jpg')

    # Define combinations
    combinations = [
        ("Original Photo", []),
        ("Correct Orientation + Crop + Upscale", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
        ]),
        ("Correct Orientation + Crop + Upscale + Enhance White Balance", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
            (enhance_white_balance, [upscale_path, white_balance_path]),
        ]),
        ("Correct Orientation + Crop + Upscale + Enhance Brightness", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
            (enhance_brightness, [upscale_path, brightness_path]),
        ]),
        ("Correct Orientation + Crop + Upscale + Enhance Contrast", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
            (enhance_contrast, [upscale_path, contrast_path]),
        ]),
        ("Correct Orientation + Crop + Upscale + Enhance White Balance + Sharpness", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
            (enhance_white_balance, [upscale_path, white_balance_path]),
            (enhance_sharpness, [white_balance_path, sharpened_path]),
        ]),
        ("Correct Orientation + Crop + Upscale + Enhance Brightness + Sharpness", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
            (enhance_brightness, [upscale_path, brightness_path]),
            (enhance_sharpness, [brightness_path, sharpened_path]),
        ]),
        ("Correct Orientation + Crop + Upscale + Enhance Contrast + Sharpness", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
            (enhance_contrast, [upscale_path, contrast_path]),
            (enhance_sharpness, [contrast_path, sharpened_path]),
        ]),
        ("Correct Orientation + Crop + Upscale + Enhance White Balance + Brightness + Contrast + Sharpness", [
            (correct_orientation, [passport_path, cropped_path]),
            (crop_based_on_eyes, [cropped_path, cropped_path]),
            (upscale, [cropped_path, upscale_path]),
            (enhance_white_balance, [upscale_path, white_balance_path]),
            (enhance_brightness, [white_balance_path, brightness_path]),
            (enhance_contrast, [brightness_path, contrast_path]),
            (enhance_sharpness, [contrast_path, sharpened_path]),
        ]),
    ]

    # Log results
    results = []

    for combo_name, steps in combinations:
        try:
            last_output_path = passport_path  # Default for combinations with no steps
            for func, args in steps:
                func(*args)
                last_output_path = args[-1]  # Update to the last argument (output path)

            # Upload the image to Google Drive and get the public URL
            image_url = upload_to_drive(last_output_path, drive_folder_id, drive_service)

            # Add result to the list without leading single quote
            results.append([combo_name, last_output_path, f'=IMAGE("{image_url}")'])

        except Exception as e:
            # Log errors
            results.append([combo_name, f"Error: {e}", ""])

    # Update Google Sheet
    values = [["Combination", "Output Path", "Image"]] + results
    body = {"values": values}
    sheet.values().update(
        spreadsheetId=sheet_id,
        range="Sheet1!A1",  # Replace 'Sheet1' with your actual sheet name
        valueInputOption="USER_ENTERED",
        body=body
    ).execute()

    print("Results logged to Google Sheet.")

def enhance_mrz(input_path, output_path, mrz_height_percentage=100):

    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")

    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate MRZ region
    mrz_start = int(height * (100 - mrz_height_percentage) / 100)
    
    # Create a mask for the MRZ region
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[mrz_start:] = 255
    
    # Extract the MRZ region
    mrz_region = image[mrz_start:, :]
    
    # Convert MRZ region to grayscale
    gray_mrz = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to handle varying lighting conditions
    binary_mrz = cv2.adaptiveThreshold(
        gray_mrz,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )
    
    # Create enhanced MRZ region with white background
    enhanced_mrz = np.full_like(mrz_region, 255)
    enhanced_mrz[binary_mrz == 0] = [0, 0, 0]  # Set text to black
    
    # Combine original image with enhanced MRZ region
    result = image.copy()
    result[mrz_start:] = enhanced_mrz
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the enhanced image
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
def enhance_colored_mrz(input_path, output_path, mrz_height_percentage=100):
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate MRZ region
    mrz_start = int(height * (100 - mrz_height_percentage) / 100)

    # Extract the MRZ region
    mrz_region = image[mrz_start:, :]

    # Convert MRZ region to grayscale
    gray_mrz = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to handle varying lighting conditions
    binary_mrz = cv2.adaptiveThreshold(
        gray_mrz,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10
    )

    # Create an empty mask for the MRZ
    binary_mrz_colored = cv2.cvtColor(binary_mrz, cv2.COLOR_GRAY2BGR)

    # Blend the original MRZ region with the enhanced version
    enhanced_colored_mrz = np.where(binary_mrz_colored == [255, 255, 255], mrz_region, [0, 0, 0])

    # Combine the enhanced MRZ with the original image
    result = image.copy()
    result[mrz_start:] = enhanced_colored_mrz

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the enhanced image
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 100])

    print(f"Enhanced image saved at {output_path}")


def enhance_photo(image_path, output_path="photo_ph.jpg", with_ai = False):
    return 1
# # folder = "chekri/P7038857C"
# folder = "mrz"
# # # process_image_combinations(folder)
# passport_path = os.path.join(folder, 'passport.jpg')
# mrz_path = os.path.join(folder, 'mrz.jpg')
# enhance_colored_mrz(passport_path, mrz_path)

# cropped_path = os.path.join(folder, 'crop.jpg')
# upscale_path = os.path.join(folder, 'upscale.jpg')
# white_balance_path = os.path.join(folder, 'white_balance.jpg')
# brightness_path = os.path.join(folder, 'brightness.jpg')
# contrast_path = os.path.join(folder, 'contrast.jpg')
# sharpened_path = os.path.join(folder, 'sharpened.jpg')
# dpi_path = os.path.join(folder, 'dpi.jpg')

# correct_orientation(passport_path, cropped_path)
# crop_based_on_eyes(cropped_path, cropped_path)
# upscale(cropped_path, upscale_path)
# enhance_white_balance(upscale_path, white_balance_path)
# # enhance_brightness(white_balance_path, brightness_path)
# enhance_contrast(white_balance_path, contrast_path)
# enhance_sharpness(contrast_path, sharpened_path)
# compress_image_set_dpi_save(sharpened_path, dpi_path)

# correct_orientation("passport.jpg", "rotated.jpg")
# crop_based_on_eyes("rotated.jpg", "crop.jpg")
# # upscale("crop.jpg", "upscale.jpg")
# enhance_contrast("crop.jpg", "contrast.jpg")
# enhance_contrast("contrast.jpg", "contrast.jpg")
# # enhance_sharpness("contrast.jpg", "sharpness.jpg")
# compress_image_set_dpi_save("contrast.jpg", "compress.jpg")