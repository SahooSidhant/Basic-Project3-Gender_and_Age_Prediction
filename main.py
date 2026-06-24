import cv2
import os

from src.detector import detect_faces
from src.detector import predict_age_gender


# -----------------------------
# IMAGE PATH
# -----------------------------

image_path = "man2.jpg"

# -----------------------------
# READ IMAGE
# -----------------------------

image = cv2.imread(image_path)

if image is None:
    print(f"Could not read image: {image_path}")
    exit()

frame = image.copy()

# -----------------------------
# DETECT FACES
# -----------------------------

faces = detect_faces(frame)

print(f"\nFaces Detected: {len(faces)}")

if not faces:
    print("No face detected.")

# -----------------------------
# PROCESS EACH FACE
# -----------------------------

for face_num, (x1, y1, x2, y2) in enumerate(faces, start=1):

    padding = 20

    face = frame[
        max(0, y1 - padding):min(y2 + padding, frame.shape[0]),
        max(0, x1 - padding):min(x2 + padding, frame.shape[1])
    ]

    if face.size == 0:
        continue

    gender, age = predict_age_gender(face)

    print(f"\nFace #{face_num}")
    print("-" * 40)
    print(f"Gender : {gender}")
    print(f"Age    : {age}")

    label = f"{gender} | {age}"

    # Draw Face Box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Draw Label
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

# -----------------------------
# SAVE OUTPUT
# -----------------------------

os.makedirs("output", exist_ok=True)

filename = os.path.splitext(
    os.path.basename(image_path)
)[0]

output_path = (
    f"output/{filename}_age_gender_result.jpg"
)

cv2.imwrite(output_path, frame)

print(f"\nResult saved: {output_path}")

# -----------------------------
# DISPLAY RESULT
# -----------------------------

cv2.imshow("Age & Gender Detection", frame)

cv2.waitKey(0)

cv2.destroyAllWindows()