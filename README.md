#DocScanner

A Python-based **Document Scanner with OCR (Optical Character Recognition)** that detects document edges from an image or webcam feed, applies perspective correction, enhances the image, and extracts text using **Tesseract OCR**.

This project simulates the functionality of a real document scanner and converts physical documents into editable digital text.


##Features

- Scan documents from an **image file**
- Scan documents using **live webcam**
- Automatic **document edge detection**
- **Perspective correction** for top-down scanned view
- Image enhancement for better OCR accuracy
- Extract text using **Tesseract OCR**
- Save extracted text to a `.txt` file
- Live preview with contour detection


##Tech Stack

- **Python**
- **OpenCV**
- **NumPy**
- **PyTesseract**
- **Pillow**
- **Imutils**


##Requirements

Install the required Python packages:

```bash
pip install opencv-python numpy pytesseract Pillow imutils
```

##Install Tesseract OCR

You must install the **Tesseract OCR engine** separately.

###Ubuntu / Debian
```bash
sudo apt install tesseract-ocr
```

###Windows
Download from: 
https://github.com/UB-Mannheim/tesseract/wiki

After installation, set the path inside the code if needed:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

###macOS
```bash
brew install tesseract
```

##Usage

###Scan from Image
```bash
python doc_scanner.py --image path/to/photo.jpg
```

###Scan from Webcam
```bash
python doc_scanner.py --webcam
```

###Save Extracted Text
```bash
python doc_scanner.py --image photo.jpg --output result.txt
```


## Main Functions

### `preprocess_for_edge(image)`
- Converts image to grayscale
- Applies Gaussian blur
- Detects edges using Canny
- Dilates edges for better contour detection

### `find_document_contour(edged)`
- Finds the largest 4-sided contour
- Assumes it is the document boundary

### `warp_document(image, contour)`
- Applies perspective transformation
- Produces a flat scanned image

### `enhance_for_ocr(image)`
- Resizes image
- Removes noise
- Applies thresholding
- Sharpens image

### `extract_text(enhanced_img)`
- Uses Tesseract OCR to extract text


