import streamlit as st
import cv2
import easyocr
import pandas as pd
from ultralytics import YOLO
import tempfile
import os

# ======================
# Allowed List
# ======================
districts = [
    "ঢাকা","ঢাকা মেট্রো","টাংগাইল","চট্টগাম","চট্র মেট্রো","খুলনা","খুলনা মেট্রো",
    "বরিশাল","বরিশাল মেট্রো","কক্সবাজার","নেত্রকোণা","রংপুর","রাজ মেট্রো","ভোলা",
    "রাজশাহী","কুষ্টিয়া","নারায়ণগঞ্জ","বগুড়া","সিরাজগঞ্জ","কুমিল্লা","ময়মনসিংহ",
    "ঝিনাইদহ","সিলেট","হবিগঞ্জ","নাটোর","পাবনা","যোশর","বরগুনা","নীলফামারী",
    "পটুয়াখালী","জামালপুর","পিরোজপুর","ব্রাক্ষণবাড়িয়া","মানিকগঞ্জ","নোয়াখালী",
    "বাগেরহাট","সুনামগঞ্জ","চুয়াডাংগা","গোপালগঞ্জ","পঞ্চগড়","লক্ষীপুর","শেরপুর",
    "ঝালকাঠি","খাগড়াছড়ি","কিশোরগঞ্জ","সাতক্ষীরা","নরসিংদী","মৌলভীবাজার","কড়িগ্রাম",
    "শড়িয়তপুর","মাদারীপুর","গাইবান্ধা","রাজবাড়ী","নওয়াবগঞ্জ","রাঙ্গামাটি","চুয়াডাঙ্গা",
    "মুন্সীগঞ্জ","নওগাঁ","গাজীপুর","মেহেরপুর","চাঁপাইনবাবগঞ্জ","বান্দরবান","চাঁদপুর",
    "জয়পুরহাট","নড়াইল","ফরিদপুর","ঠাকুরগাঁও","লালমনিরহাট"
]

allowed_chars = [
    "গ","হ","ল","ঘ","চ","ট","থ","এ",
    "ক","খ","ভ","প","ছ","জ","ঝ","ব",
    "স","ত","দ","ফ","ঠ","ম","ন","অ",
    "ড","উ","ঢ","শ","ই","য","র"
]

allowed_digits = [
    "০","১","২","৩","৪","৫","৬","৭","৮","৯"
]

def filter_text(text: str) -> bool:
    text = text.strip()
    if text in districts:   # Full district match
        return True
    if text in allowed_chars:   # Single allowed char
        return True
    if all(ch in allowed_digits for ch in text):   # Digits only
        return True
    return False

# ======================
# Streamlit UI
# ======================
st.set_page_config(page_title="Bangla Number Plate OCR", layout="wide")
st.title("📑 Bangla Number Plate OCR Detection")

col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### ⚙️ Options")
    frame_skip = st.slider("Frame Skip (Video)", 1, 30, 10)
    show_raw = st.checkbox("Show Raw OCR Results (Unfiltered)", False)

# Load YOLO model
model = YOLO("best.pt")

# EasyOCR reader
reader = easyocr.Reader(['en', 'bn'])

# Upload file
uploaded_file = st.file_uploader("📤 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mkv"])

if uploaded_file:
    temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    results_list = []
    unique_detected = set()   # ✅ prevent duplicate

    # ----------------------
    # Image Processing
    # ----------------------
    if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
        with col1:
            st.image(temp_path, caption="Uploaded Image", use_column_width=True)

        results = model(temp_path)
        img = cv2.imread(temp_path)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                roi = img[y1:y2, x1:x2]
                ocr_result = reader.readtext(roi)
                for (_, text, _) in ocr_result:
                    text = text.strip()
                    if filter_text(text):
                        if text not in unique_detected:
                            results_list.append([text])
                            unique_detected.add(text)
                    elif show_raw:
                        if text not in unique_detected:
                            results_list.append([text])
                            unique_detected.add(text)

    # ----------------------
    # Video Processing
    # ----------------------
    elif uploaded_file.name.lower().endswith((".mp4", ".avi", ".mkv")):
        with col1:
            st.video(temp_path)

        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip == 0:  # প্রতি n ফ্রেমে OCR
                results = model(frame)
                for r in results:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        roi = frame[y1:y2, x1:x2]
                        ocr_result = reader.readtext(roi)
                        for (_, text, _) in ocr_result:
                            text = text.strip()
                            if filter_text(text):
                                if text not in unique_detected:   # ✅ একবারই detect হবে
                                    results_list.append([text])
                                    unique_detected.add(text)
                            elif show_raw:
                                if text not in unique_detected:
                                    results_list.append([text])
                                    unique_detected.add(text)
        cap.release()

    # ----------------------
    # Show Results
    # ----------------------
    if results_list:
        df = pd.DataFrame(results_list, columns=["Detected Text"])
        st.subheader("📌 OCR Results")
        st.dataframe(df)

        csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ Download CSV", csv, "ocr_results.csv", "text/csv")
    else:
        st.warning("⚠️ No valid text detected!")
