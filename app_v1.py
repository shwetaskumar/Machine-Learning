import streamlit as st
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,WebRtcMode
import av
import cv2
import numpy as np
import cvlib as cv
from PIL import Image
from keras.models import load_model

st.title("PREDICT BMI, GENDER AND AGE")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
model = load_model('best_bmi_model_v3_3.h5', compile=False)
age_model = load_model('best_age_model_v1_2.h5', compile=False)
gender_model = load_model('best_gender_model_v1_0.h5')

def image_resize(image, size):
    # Get the dimensions of the image
    height, width, _ = image.shape
    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Determine the resizing dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    elif aspect_ratio == 1:
        new_width = size
        new_height = size
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    # Resize the image using the determined dimensions
    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a black canvas of the desired size
    padded_image = np.zeros((size, size, 3), dtype=np.uint8)

    # Calculate the padding values
    pad_top = (size - new_height) // 2
    pad_left = (size - new_width) // 2

    # Copy the resized image onto the canvas with padding
    padded_image[pad_top:pad_top+new_height, pad_left:pad_left+new_width] = resized_image

    return padded_image

def process_face(face):
    image = image_resize(face, size=224)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.copy(image)
    image = image / 255.0
    return image

def process_box(frame, f):
    x, y = f[0], f[1]
    x2, y2 = f[2], f[3]
    x = x-100
    x2 = x2+100
    y = y-100
    y2 = y2+100
    x = 0 if x < 1 else x
    y = 0 if y < 1 else y
    face_select = frame[y:y2, x:x2]
    face_select = process_face(face_select)
    return face_select

class VideoProcessor:
    def recv(self, frame):
        gender_labels = ["Female", "Male"]
        img = frame.to_ndarray(format="bgr24")

        # Detect faces in the frame using cvlib
        faces, _ = cv.detect_face(img)
        
        # Prepare faces for processing
        face_images = []
        for idx, i in enumerate(faces):
            x, y, w, h = i
            cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
            face_images.append(process_box(img, i))

        try:
            if len(face_images) > 0:
                face_images = np.array(face_images, dtype="float32")
                prediction = model.predict(face_images, batch_size=32)
                gender_prediction = gender_model.predict(face_images, batch_size=32)
                age_prediction = age_model.predict(face_images, batch_size=32)

                for (x, y, x2, y2), bmi_value, gender_pred, age_pred in zip(faces, prediction, gender_prediction, age_prediction):
                    gender_pred_label = gender_labels[np.argmax(gender_pred)]

                    cv2.putText(img, f'BMI: {bmi_value[0]:.2f}', (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(img, f'Gender: {gender_pred_label}', (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(img, f'Age: {round(age_pred[0])}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        except Exception as e:
            print(e)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def tab1():
    st.title("Webcam")
    webrtc_streamer(key="webcam", mode=WebRtcMode.SENDRECV, video_processor_factory=VideoProcessor, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"video": True, "audio": False}, async_processing=True)

def tab2():
    st.title("Upload Photo")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # Process the image here
        gender_labels = ["Female", "Male"]
        # Detect faces in the frame using cvlib
        faces, _ = cv.detect_face(image)
        
        # Prepare faces for processing
        face_images = []
        for idx, i in enumerate(faces):
            x, y, w, h = i
            cv2.rectangle(image, (x, y), (w, h), (255, 0, 0), 2)
            face_images.append(process_box(image, i))

        try:
            if len(face_images) > 0:
                face_images = np.array(face_images, dtype="float32")
                prediction = model.predict(face_images, batch_size=32)
                gender_prediction = gender_model.predict(face_images, batch_size=32)
                age_prediction = age_model.predict(face_images, batch_size=32)

                for (x, y, x2, y2), bmi_value, gender_pred, age_pred in zip(faces, prediction, gender_prediction, age_prediction):
                    gender_pred_label = gender_labels[np.argmax(gender_pred)]

                    cv2.putText(image, f'BMI: {bmi_value[0]:.2f}', (x, y-80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                    cv2.putText(image, f'Gender: {gender_pred_label}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                    cv2.putText(image, f'Age: {round(age_pred[0])}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

                # Display the updated image with bounding boxes and predictions
                st.image(image, caption="Processed Image", use_column_width=True)
        except Exception as e:
            print(e)


def main():
    # Create the tabs
    tabs = ["Webcam", "Upload Photo"]
    selected_tab = st.selectbox("Select a tab", tabs)

    # Run the selected tab's function
    if selected_tab == "Webcam":
        tab1()
    elif selected_tab == "Upload Photo":
        tab2()

if __name__ == "__main__":
    main()
