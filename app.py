import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your model (replace with your model path)
model = load_model('_model.h5')

# Define the labels for your Ayurvedic plants
# You should replace this list with the actual labels that your model uses
plant_labels = ["Crape Jasmine", "Curry", "Guava", "Lemon", "Mango"]  # Modify with your actual plant names

# Preprocess the uploaded image to match the model's input requirements
def preprocess_image(img):
    img = img.convert('RGB')  # Convert to RGB if the image is in another mode
    img = img.resize((400, 400))  # Resize to match your model's expected input size (e.g., 224x224)
    img_array = np.array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image if your model was trained with normalized data
    return img_array

# Define the prediction function
def predict_plant(img):
    img_array = preprocess_image(img)  # Preprocess the image
    predictions = model.predict(img_array)  # Predict using the model
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    plant_name = plant_labels[predicted_class[0]]  # Get the plant name from the label list
    confidence = np.max(predictions)  # Get the confidence of the prediction
    return plant_name, confidence

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_plant,  # Function to call
    inputs=gr.Image(type="pil", label="Upload an Image of the Plant"),  # Input type is an image
    outputs=[gr.Textbox(label="Predicted Plant Name"), gr.Textbox(label="Confidence Level")],  # Outputs predicted label and confidence
    live=True  # Enable real-time interaction
)

# Launch the interface
iface.launch()
