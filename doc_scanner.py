import cv2
import numpy as np
import pytesseract
import argparse
import sys
from PIL import Image
from imutils.perspective import four_point_transform
import imutils

def preprocess_for_edge(image):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged   = cv2.Canny(blurred, 30, 100)
    # Dilate so contours close up
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged   = cv2.dilate(edged, kernel, iterations=1)
    return edged


def find_document_contour(edged):
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    for c in cnts:
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx
    return None


def warp_document(image, contour):
    return four_point_transform(image, contour.reshape(4, 2))


def enhance_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale if too small (Tesseract works better on larger images)
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1000 / max(h, w)
        gray  = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold → clean black-on-white
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 10
    )

    # Mild sharpening
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1,  0]])
    sharp  = cv2.filter2D(thresh, -1, kernel)
    return sharp

def extract_text(enhanced_img):
    pil_img = Image.fromarray(enhanced_img)
    config  = "--oem 3 --psm 6"   # assume uniform text block; change psm if needed
    text    = pytesseract.image_to_string(pil_img, config=config)
    return text.strip()


def draw_contour_overlay(image, contour):
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 100), 3)
    pts = contour.reshape(4, 2).astype(int)
    for pt in pts:
        cv2.circle(overlay, tuple(pt), 10, (0, 80, 255), -1)
    return overlay


def show_results(original, warped, enhanced, text):
    # Resize for display
    disp_h = 700
    def resize_h(img, h):
        r = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * r), h))

    left  = resize_h(warped, disp_h)
    right_gray = resize_h(enhanced, disp_h)
    right = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2BGR)

    # Text panel
    text_panel = np.zeros((disp_h, 500, 3), dtype=np.uint8)
    text_panel[:] = (25, 25, 35)

    cv2.putText(text_panel, "EXTRACTED TEXT", (12, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (100, 200, 255), 1)
    cv2.line(text_panel, (10, 42), (490, 42), (60, 60, 80), 1)

    y = 68
    for line in text.split("\n"):
        # Word-wrap at ~55 chars
        while len(line) > 55:
            cv2.putText(text_panel, line[:55], (12, y),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 220, 220), 1)
            line = line[55:]
            y += 20
        cv2.putText(text_panel, line, (12, y),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (220, 220, 220), 1)
        y += 20
        if y > disp_h - 20:
            break

    # Stack horizontally
    combined = np.hstack([left, right, text_panel])
    cv2.imshow("Document Scanner  |  Warped  |  Enhanced  |  Text", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scan_image(image):
    ratio  = image.shape[0] / 500.0
    resized = imutils.resize(image, height=500)

    edged   = preprocess_for_edge(resized)
    contour = find_document_contour(edged)

    if contour is not None:
        print("[✓] Document edges detected – applying perspective correction.")
        # Scale contour back to original size
        contour_full = (contour * ratio).astype(np.float32)
        warped       = warp_document(image, contour_full)
        # Show outline on resized for debug (optional)
        draw_contour_overlay(resized, contour)
    else:
        print("[!] No clear document edge found – using full image.")
        warped = image

    enhanced = enhance_for_ocr(warped)
    text     = extract_text(enhanced)
    return warped, enhanced, text

def run_image(path, output_path=None):
    image = cv2.imread(path)
    if image is None:
        print(f"[✗] Could not open image: {path}")
        sys.exit(1)

    print(f"[•] Processing {path} ...")
    warped, enhanced, text = scan_image(image)

    print("\n" + "═" * 60)
    print("EXTRACTED TEXT")
    print("═" * 60)
    print(text)
    print("═" * 60 + "\n")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[✓] Text saved to {output_path}")

    show_results(image, warped, enhanced, text)


def run_webcam(output_path=None):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[•] Webcam mode  –  press  S  to scan,  Q  to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Live edge preview
        ratio   = frame.shape[0] / 500.0
        resized = imutils.resize(frame, height=500)
        edged   = preprocess_for_edge(resized)
        contour = find_document_contour(edged)

        preview = resized.copy()
        if contour is not None:
            preview = draw_contour_overlay(resized, contour)
            cv2.putText(preview, "Document detected – press S to scan",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 100), 2)
        else:
            cv2.putText(preview, "Point camera at a document",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 180, 255), 2)

        cv2.imshow("Document Scanner – Live", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            print("[•] Captured – scanning ...")
            warped, enhanced, text = scan_image(frame)

            print("\n" + "═" * 60)
            print("EXTRACTED TEXT")
            print("═" * 60)
            print(text)
            print("═" * 60 + "\n")

            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"[✓] Text saved to {output_path}")

            show_results(frame, warped, enhanced, text)
            return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Scanner + OCR")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  help="Path to an image file")
    group.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--output", help="Save extracted text to this file")
    args = parser.parse_args()

    if args.image:
        run_image(args.image, args.output)
    else:
        run_webcam(args.output)
