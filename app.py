import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("👤 Face Detection App using Viola–Jones Algorithm")

# --- 1. Load the Face Detector (Moved to top to fix NameError) ---
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    st.error("Error loading face cascade. Make sure opencv-python is installed correctly.")

# --- Instructions ---
st.write("""
### Instructions:
1. Upload an image (JPG, JPEG, or PNG).
2. Choose the color of the rectangle for detected faces.
3. Adjust the `scaleFactor` and `minNeighbors` parameters.
4. The app will automatically detect faces.
5. Click **Save Image** to download the result.
""")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Fixed: changed use_column_width to use_container_width
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # --- User Inputs ---
    st.markdown("### Detection Settings")
    color = st.color_picker("Choose rectangle color", "#00FF00")
    
    # Scale factor must be > 1.0
    scale_factor = st.slider("Adjust scaleFactor", 1.01, 2.0, 1.1, 0.01)
    min_neighbors = st.slider("Adjust minNeighbors", 1, 10, 5)

    # Convert hex color to RGB tuple
    rect_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    # --- Processing ---
    if face_cascade.empty():
        st.error("XML file not loaded. Check your OpenCV installation.")
    else:
        img_processed = img_rgb.copy()
        faces = face_cascade.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(img_processed, (x, y), (x + w, y + h), rect_color, 2)

        st.image(img_processed, caption=f"Detected Faces: {len(faces)}", use_container_width=True)

        # --- Save the image ---
        result_image = Image.fromarray(img_processed)
        
        buf = io.BytesIO()
        result_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="Save Image",
            data=byte_im,
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )