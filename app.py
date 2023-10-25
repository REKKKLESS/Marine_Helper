from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  # Add these imports
import numpy as np
import io 
import tensorflow as tf




app = Flask(__name__)
app.static_url_path = '/static'
@app.route('/')
def index():
    return render_template('index.html')
# Load the pre-trained model
model = load_model("Rohitfinal.h5")

# Create an ImageDataGenerator for preprocessing
testGen = ImageDataGenerator(preprocessing_function=preprocess_input)

import io

model_labels = {
        0: 'Black Sea Sprat',
        1 :'Gilt-Head Bream',
        2 :'Hourse Mackerel',
        3 :'Red Mullet',
        4 :'Red Sea Bream',
        5 :'Sea Bass',
        6 :'Shrimp',
        7 :'Striped Red Mullet',
        8 :'Trout'
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        file = request.files['image']
        
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        
        # Read image data from the FileStorage object
        image_data = file.read()
        
        # Convert the image data into a bytes-like object
        image_bytes = io.BytesIO(image_data)
        
        
        
        
        # Load and preprocess the image
        img = load_img(image_bytes, target_size=(256, 256))  # Use load_img from keras.preprocessing.image
        img = tf.convert_to_tensor(img)  # Convert to a TensorFlow tensor

        # Extract the first 3 color channels (R, G, B)
        img = img[:,:,:3]
        img_array = img_to_array(img)  # Convert to NumPy array
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Preprocess the image
        
        # Make predictions using the model
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions, axis=1)[0]
        predicted_class = model_labels.get(predicted_label, 'Unknown')
        
        return jsonify({'predicted_class': predicted_class}), 200
    
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 400
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/home.html')
def home():
    return render_template('home.html')


@app.route('/species.html')
def species():
    return render_template('species.html')
if __name__ == "__main__":
    app.run(debug=True)
