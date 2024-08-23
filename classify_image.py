import numpy as np
from PIL import Image
from keras.models import load_model

# Load the pre-trained model
model = load_model('trained_model.h5')

# Define a function to preprocess the user-provided image
def preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Convert the image to grayscale if it's not already in grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize the image to match the input size of the pre-trained model
    img = img.resize((224, 224))
    
    # Convert the image to a numpy array and normalize pixel values
    img_array = np.array(img) / 255.0
    
    # Stack the image to have 3 channels (RGB)
    img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Define a function to classify the user-provided image
def classify_image(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Use the pre-trained model to make predictions
    predictions = model.predict(processed_image)
    
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    
    # Map the predicted class index to the actual class label (e.g., using a dictionary)
    class_labels = {0: 'healthy: not DR', 1: 'mild DR', 2: 'moderate DR', 3: 'Proliferative DR', 4: 'severe DR'}
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

# Take input image path from the user
image_path = input("Enter the path to the image you want to classify: ")

# Classify the user-provided image
predicted_class = classify_image(image_path)

# Display the predicted class to the user
print("Predicted Class:", predicted_class)
