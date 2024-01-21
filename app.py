from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import joblib
import openai

# Create a Flask app
app = Flask(__name__)

# Load the trained model using joblib
model = joblib.load('cnn_terrain_detection_18_09.pkl')

# Set your OpenAI API key
openai.api_key = "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"  # Replace with your actual OpenAI API key

# Define a function to preprocess an image
def preprocess_image(image, target_size):
    # Load and resize the image
    img = tf.image.decode_image(image)
    img = tf.image.resize(img, [target_size, target_size])
    img = img / 255.0  # Normalize image
    return img.numpy()

# Define a route to upload an image and get the predicted class name and curing techniques
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the request
        image = request.files['image'].read()

        # Preprocess the image
        processed_image = preprocess_image(image, target_size=256)

        # Make predictions using the model
        predictions = model.predict(np.expand_dims(processed_image, axis=0))

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Define class labels (modify as needed)
        class_labels = ["Acne and Rosacea", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions", "Atopic Dermatitis", "Bullous Disease"]

        # Get the class name corresponding to the predicted class
        predicted_class_name = class_labels[predicted_class_index]

        # Generate a prompt for OpenAI with the predicted class name
        openai_prompt = f"Please provide information on {predicted_class_name} disease, its symptoms, and treatment options for it, depending on the severity of the disease in 5 points each. Each point should be on a new line and in layman's terms."

        # Use OpenAI API to get a response
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose the appropriate OpenAI engine
            prompt=openai_prompt,
            max_tokens=500,  # Adjust the max tokens as needed
        )

        # Extract the text from the OpenAI response
        openai_response = response.choices[0].text

        # Combine the predicted class name and OpenAI response
        result = f"Predicted Disease: <b>{predicted_class_name}</b><br><br> Symptoms and Treatment:<br>{openai_response}"

        return result
    except Exception as e:
        return "Error: " + str(e)

# Define a route to render an HTML form for image upload


# Define a route to render an HTML form for image upload

@app.route('/')
def sih():
    return render_template('sih.html')

# Define a route to render the index.html page
@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
