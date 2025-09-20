# omr/pipeline.py
import cv2
import numpy as np
from PIL import Image
import base64
import io
from .utils import CANON_WIDTH, CANON_HEIGHT, generate_grid_cells

def read_image_bytes(file_bytes):
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def preprocess_image(img):
    # Resize large images for speed (while keeping aspect)
    h, w = img.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

def find_document_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def order_points(pts):
    # from imutils - order tl,tr,br,bl
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_document(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],[CANON_WIDTH-1,0],[CANON_WIDTH-1,CANON_HEIGHT-1],[0,CANON_HEIGHT-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (CANON_WIDTH, CANON_HEIGHT))
    return warped

def threshold_for_bubbles(gray):
    # adaptive threshold + morphological ops
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV,15,8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return th

def evaluate_cells(warped_color):
    gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    th = threshold_for_bubbles(gray)
    cells = generate_grid_cells()
    results = []  # list of dict {q_no, filled_ratio, marked(Boolean)}
    overlay = warped_color.copy()
    for idx, (x,y,w,h) in enumerate(cells):
        cell = th[y:y+h, x:x+w]
        total = w*h
        filled = np.count_nonzero(cell)
        filled_ratio = filled / float(total)
        # thresholds tuned for prototype:
        marked = filled_ratio > 0.18
        # draw rect on overlay
        color = (0,255,0) if marked else (0,0,255)
        cv2.rectangle(overlay, (x,y),(x+w,y+h), color, 2)
        results.append({"q": idx+1, "filled_ratio": float(filled_ratio), "marked": bool(marked)})
    return results, overlay

def mark_to_answer(results):
    # Each question in prototype is single-choice A/B/C/D mapped by quadrant inside cell.
    # To keep simple, we assume each cell is split horizontally into 4 equal subcells left->right for options A-D.
    # More robust systems compute per-option subcell; here we return 'A' if marked True for that question.
    answers = {}
    for r in results:
        # simple rule: if marked => choose 'A' (prototype)
        answers[str(r["q"])] = "A" if r["marked"] else ""
    return answers

def score_answers(extracted_answers, answer_key):
    # answer_key: dict question_no(str) -> correct option e.g. "A"
    per_subject = [0]*5
    total = 0
    wrong = []
    for q_str, correct in answer_key.items():
        q = int(q_str)
        extracted = extracted_answers.get(q_str, "")
        if extracted == correct:
            # determine subject index: 0..4
            subj = (q-1) // 20
            per_subject[subj] += 1
            total += 1
        else:
            wrong.append(q)
    return per_subject, total, wrong

def pil_image_to_bytes(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

