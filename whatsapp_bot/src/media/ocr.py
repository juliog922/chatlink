import logging
import numpy as np
import cv2
from paddleocr import PaddleOCR

# Initialize only once globally
ocr_model = PaddleOCR(lang="es", use_angle_cls=True, show_log=False)


def binarize_and_normalize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def extract_text_from_image(image_bytes: bytes) -> str:
    try:
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        image = binarize_and_normalize(image)
        result = ocr_model.ocr(image, cls=True)

        extracted_text = [line[1][0] for block in result for line in block]
        return "\n".join(extracted_text).strip()
    except Exception as e:
        logging.error(f"OCR error: {e}")
        return ""
