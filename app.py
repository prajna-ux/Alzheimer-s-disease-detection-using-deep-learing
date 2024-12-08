from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Flask app setup
app = Flask(__name__)
app.secret_key = "your_secret_key"
app.config["MONGO_URI"] = "mongodb://localhost:27017/alzheimers_app"
mongo = PyMongo(app)
users_collection = mongo.db.users  # MongoDB collection for user data

# Load environment variables from a .env file
load_dotenv()

# Gemini API key configuration
genai_api_key = os.getenv("GENAI_API_KEY")
if not genai_api_key:
    logging.error("API Key for Gemini is not configured. Please set GENAI_API_KEY in environment variables.")
    raise ValueError("API Key not found")

genai.configure(api_key=genai_api_key)

# Session lifetime
app.permanent_session_lifetime = timedelta(minutes=30)
if os.getenv("ENV") == "production":
    app.config['SESSION_COOKIE_SAMESITE'] = 'None'
    app.config['SESSION_COOKIE_SECURE'] = True

# Load the pre-trained CNN model for Alzheimer's prediction
#cnn_model = load_model('model_CNN.h5')


# Load both pre-trained models
cnn_model = load_model('model_CNN.h5')
resnet_model = load_model('resnet50_model.h5')  # Load your ResNet model

# Class names for the model predictions
class_names = ["No Dementia", "Mild Dementia", "Moderate Dementia", "Very Mild Dementia"]

# Ensure the uploads folder exists
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Class names for the model predictions (assuming both models use the same classes)
class_names = ["No Dementia", "Mild Dementia", "Moderate Dementia", "Very Mild Dementia"]

# Update preprocess_image function to handle both models
def preprocess_image(filepath):
    """Preprocess the image for prediction."""
    image = load_img(filepath, target_size=(224, 224))  # Both CNN and ResNet commonly use 224x224
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array /= 255.0
    return image_array
# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        middle_name = request.form.get('middle_name')
        last_name = request.form.get('last_name')
        gender = request.form.get('gender')
        dob = request.form.get('dob')
        address = request.form.get('address')
        place = request.form.get('place')
        district = request.form.get('district')
        state = request.form.get('state')
        phone_number = request.form.get('phone_number')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username already exists
        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            flash("User already exists! Please log in.", "warning")
            return redirect(url_for('login'))

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "first_name": first_name,
            "middle_name": middle_name,
            "last_name": last_name,
            "gender": gender,
            "dob": dob,
            "address": address,
            "place": place,
            "district": district,
            "state": state,
            "phone_number": phone_number,
            "email": email,
            "username": username,
            "password": hashed_password
        })

        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            session.permanent = True
            return redirect(url_for('main'))
        else:
            flash("Invalid credentials! Please try again.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    if 'username' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part!", "warning")
            return redirect(url_for('main'))

        file = request.files['file']
        if file.filename == '':
            flash("No selected file!", "warning")
            return redirect(url_for('main'))

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess the image
            image_array = preprocess_image(filepath)

            # Get predictions from both models
            cnn_predictions = cnn_model.predict(image_array)
            resnet_predictions = resnet_model.predict(image_array)

            # Combine predictions (e.g., averaging)
            combined_predictions = (cnn_predictions + resnet_predictions) / 2

            # Get the predicted class
            predicted_class = class_names[np.argmax(combined_predictions)]
            result = predicted_class

            # Store the predicted class in session
            session['predicted_class'] = predicted_class

    return render_template('main.html', username=session.get('username'), result=result)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_message = request.form.get('user_message', '').strip()
        if not user_message:
            return {"doctor_response": "Please enter a valid message."}, 400

        predicted_class = session.get('predicted_class')
        if not predicted_class:
            return {"doctor_response": "No prediction available. Please upload an image first."}, 400

        try:
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(
                f"The user has been diagnosed with {predicted_class}. Answer the user's question: {user_message}"
            )
            doctor_response = response.text
            return {"doctor_response": doctor_response}, 200
        except Exception as e:
            logging.error("Error using Gemini API: %s", e)
            return {"doctor_response": "An unexpected error occurred while processing your request."}, 500

    # For GET requests or direct access to the page
    return render_template('chat.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)