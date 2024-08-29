from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from io import BytesIO
import os

app = Flask(__name__)

def extract_cards_from_url(image_url, output_folder):
    # Step 1: Read the Image from URL
    response = requests.get(image_url)
    image_data = BytesIO(response.content)
    image = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_COLOR)
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Preprocess the Image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Step 3: Find Card Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_count = 0
    card_files = []
    for contour in contours:
        # Step 4: Filter and Approximate Contours
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # We're looking for quadrilateral shapes
            # Check area to avoid noise
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area threshold
                continue
            
            # Check aspect ratio to filter out non-card contours
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 2.0:  # Typical card aspect ratio range
                # Step 5: Extract and Warp Perspective
                pts = approx.reshape(4, 2)
                rect = np.zeros((4, 2), dtype="float32")

                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                (tl, tr, br, bl) = rect
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))

                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))

                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))

                # Step 6: Save Individual Card Images
                card_filename = os.path.join(output_folder, f"card_{card_count}.jpg")
                cv2.imwrite(card_filename, warped)
                card_files.append(card_filename)
                card_count += 1

    return card_count, card_files

@app.route('/', methods=['GET'])
def test_endpoint():
    image_url = request.args.get('image_url')
    output_folder = './extracted_cards'
    os.makedirs(output_folder, exist_ok=True)

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        card_count, card_files = extract_cards_from_url(image_url, output_folder)
        return jsonify({
            "message": f"Extracted {card_count} cards.",
            "card_files": card_files
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
