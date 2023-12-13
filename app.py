#imports
import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model
Tumour_detect = load_model(r'C:\Users\rayen\OneDrive\Desktop\LCS3\Machine learning\ModelData')#change this to the model data path

class_labels={0: 'adenocarcinoma', 1: 'large.cell.carcinoma', 2: 'normal', 3: 'squamous.cell.carcinoma'}

treatments_dict = {
    0: "Adenocarcinoma is often treated with a combination of surgery, chemotherapy, and targeted therapy.",
    1: "Large cell carcinoma treatment may involve surgery, chemotherapy, and radiation therapy.",
    3: "Squamous cell carcinoma treatment may include surgery, radiation therapy, and immunotherapy.",
    
}
def preprocess_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array


def predict(image_path, model):

    img_array = Image.open(image_path).convert("RGB")
    img_array = img_array.resize((224, 224))
    img_array = np.array(img_array)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0) # Expand dimensions to match model's expected input shape
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return predicted_class

# Streamlit app
st.title("Tumor Detect")
uploaded_file = st.file_uploader("Choose an image...", type=("png", "jpg", "jpeg", "gif", "bmp"))

if uploaded_file is not None:
    predicted_class = predict(uploaded_file, Tumour_detect)
    # Display the results 
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    st.image(img, caption=f"Predicted: {class_labels[predicted_class]}", use_column_width=True)
    if predicted_class != 2:  # normal class doesn't need treatement
        recommended_treatments = treatments_dict.get(predicted_class, "No specific treatments available.")
        st.text("Recommended Treatments:")
        st.text(recommended_treatments)
    else:
        st.text("No specific treatments are recommended for you. Regular health check-ups are advised.")
