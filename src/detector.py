import cv2
import os


# -----------------------------
# MODEL PATHS
# -----------------------------

FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"

AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"

GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

# -----------------------------
# LABELS
# -----------------------------

AGE_LIST = [
    "(0-2)",
    "(4-6)",
    "(8-12)",
    "(15-20)",
    "(25-32)",
    "(38-43)",
    "(48-53)",
    "(60-100)"
]

GENDER_LIST = [
    "Male",
    "Female"
]

MODEL_MEAN_VALUES = (
    78.4263377603,
    87.7689143744,
    114.895847746
)

# -----------------------------
# LOAD MODELS
# -----------------------------

print("Loading models...")

faceNet = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
ageNet = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
genderNet = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

print("Models loaded successfully!")

# -----------------------------
# FACE DETECTION FUNCTION
# -----------------------------

def detect_faces(frame):

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 117, 123),
        swapRB=True,
        crop=False
    )

    faceNet.setInput(blob)

    detections = faceNet.forward()

    face_boxes = []

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)

            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            face_boxes.append((x1, y1, x2, y2))

    return face_boxes


# -----------------------------
# AGE + GENDER PREDICTION
# -----------------------------

def predict_age_gender(face):

    blob = cv2.dnn.blobFromImage(
        face,
        scalefactor=1.0,
        size=(227, 227),
        mean=MODEL_MEAN_VALUES,
        swapRB=False
    )

    # Gender Prediction
    genderNet.setInput(blob)
    gender_preds = genderNet.forward()

    gender_idx = gender_preds[0].argmax()
    gender = GENDER_LIST[gender_idx]

    # Age Prediction
    ageNet.setInput(blob)
    age_preds = ageNet.forward()

    age_idx = age_preds[0].argmax()
    age = AGE_LIST[age_idx]

    return gender, age