import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Real-Time Object Detection")
st.write("YOLOv8 Object Detection System")


@st.cache_resource
def load_model():
    return YOLO("yolov8n.onnx")


model = load_model()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    with st.spinner("Detecting objects..."):
        results = model(image)

    annotated_image = results[0].plot()

    st.image(
        annotated_image,
        caption="Detection Result",
        use_container_width=True
    )

    st.success("Detection completed!")