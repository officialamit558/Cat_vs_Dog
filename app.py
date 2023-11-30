from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained CNN model
model = load_model('cat_dog_model.h5')

# Define the class labels
class_labels = ['Cat', 'Dog']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['image']

    # Save the image to a temporary file
    img_path = 'temp_img.jpg'
    img_file.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make predictions
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Get the class label
    predicted_label = class_labels[predicted_class]

    # Render the result
    return render_template('result.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
