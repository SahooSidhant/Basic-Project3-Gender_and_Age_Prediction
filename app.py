import streamlit as st
import cv2
import numpy as np
import os

from src.detector import detect_faces
from src.detector import predict_age_gender

st.set_page_config(
    page_title="Age & Gender Detection",
    page_icon="🧑",
    layout="wide"
)

st.markdown("""
<style>

/* Main Container */
.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Background */
.stApp {
    background-color: #0E1117;
}

/* Title */
h1 {
    color: #00E5FF !important;
}

/* Detect Button */
.stButton > button {
    background-color: #00E5FF;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 66px;
    border: none;
    margin-top: 29px;
}

/* Download Button */
.stDownloadButton > button {
    background-color: #00C853;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
# 🧑 Age & Gender Detection System

Detect faces and estimate age groups and gender using
OpenCV DNN and pre-trained Deep Learning models.
""")

col1, col2 = st.columns([5, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    detect_clicked = st.button(
        "🔍 Detect",
        use_container_width=True
    )

image_placeholder = st.empty()

if uploaded_file:

    image_placeholder.image(
        uploaded_file,
        caption="Uploaded Image",
        width=600
    )

    if detect_clicked:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        faces = detect_faces(frame)

        # st.metric(
        #     "Faces Detected",
        #     len(faces)
        # )

        results = []

        for face_num, (x1, y1, x2, y2) in enumerate(faces, start=1):

            padding = 20

            face = frame[
                max(0, y1 - padding):min(y2 + padding, frame.shape[0]),
                max(0, x1 - padding):min(x2 + padding, frame.shape[1])
            ]

            if face.size == 0:
                continue

            gender, age = predict_age_gender(face)

            label = f"{gender} | {age}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            results.append({
                "Face": face_num,
                "Gender": gender,
                "Age Group": age
            })

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        filename = os.path.splitext(
            uploaded_file.name
        )[0]

        download_filename = (
            f"{filename}_age_gender_result.jpg"
        )

        _, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        image_placeholder.empty()

        left, right = st.columns([1.8, 1])

        # with left:

        #     st.image(
        #         frame_rgb,
        #         caption="Detection Result",
        #         width=500
        #     )

        # with right:

        #     st.subheader("📋 Detection Summary")

        #     st.table(results)

        #     st.download_button(
        #         label="📥 Download Result",
        #         data=buffer.tobytes(),
        #         file_name=download_filename,
        #         mime="image/jpeg"
        #         )



        with left:

            st.image(
                frame_rgb,
                width=650
            )

        with right:

            st.markdown(
                f"""
                <div style="
                    background-color:#161B22;
                    padding:20px;
                    border-radius:15px;
                ">
                <h3 style="color:#00E5FF;margin-bottom:0;">
                    📊 Faces Detected
                </h3>

                <h1 style="
                    color:white;
                    margin-top:5px;
                ">
                    {len(faces)}
                </h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("### 📋 Detection Summary")

            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True
            )

            st.download_button(
                label="📥 Download Result",
                data=buffer.tobytes(),
                file_name=download_filename,
                mime="image/jpeg",
                use_container_width=True
            )
 
st.markdown("---")

st.caption(
    "Built with OpenCV, Streamlit and Pre-trained CNN Models"
)