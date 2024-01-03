# Import necessary libraries
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load the saved model
model_path = os.path.join(os.path.dirname(__file__), 'Saved_model')  # Assuming 'app.py' is in the root directory
model = load_model(model_path)

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the form
        file = request.files['file']

        # Save the file to a temporary location
        file_path = 'temp_upload.jpg'
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make a prediction
        prediction = model.predict(img_array)

        # Get the result
        result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

        # Render the result on the web page
        return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)  # Use a different port, e.g., 8000
