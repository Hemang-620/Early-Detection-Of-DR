# Import the necessary library
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import save_model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the dataset
def prepare_dataset(dataset_dir, batch_size=32, img_size=(224, 224), validation_split=0.2):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Step 2: Define CNN model architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Step 3: Train the model on the dataset
def train_model(model, train_generator, validation_generator, epochs=10):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)
    return history

# Step 4: Evaluate the model's performance
def evaluate_model(model, validation_generator):
    loss, accuracy = model.evaluate(validation_generator)
    print(f'Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}')

    y_true = validation_generator.classes
    y_pred = np.argmax(model.predict(validation_generator), axis=-1)
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Main function
if __name__ == "__main__":
    dataset_dir = 'dataset DR'  # Add your dataset path here
    img_size = (224, 224)
    batch_size = 32
    num_classes = 5  # Assuming you have 5 classes: Healthy, Mild DR, Moderate DR, Proliferative DR, Severe DR

    # Prepare dataset
    train_generator, validation_generator = prepare_dataset(dataset_dir, batch_size=batch_size, img_size=img_size)

    # Create and train model
    model = create_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    history = train_model(model, train_generator, validation_generator)
    # Save the trained model
    model.save('trained_model.h5') # Add the path where you want to save your model here
    # Evaluate model
    evaluate_model(model, validation_generator)

