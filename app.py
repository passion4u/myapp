import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# 学習済みモデルのロード
try:
    model = load_model('banknote_classifier.h5')
    categories = ['Japan_banknote', 'USA_banknote', 'Euro_banknote', 'UK_banknote', 'China_banknote', 'Canada_banknote', 'Australia_banknote', 'India_banknote', 'Brazil_banknote', 'Russia_banknote']
except Exception as e:
    app.logger.error(f"Error loading model: {e}")
    model = None

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(50, 50))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        return categories[np.argmax(prediction)]
    except Exception as e:
        app.logger.error(f"Error predicting image: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            if prediction:
                return render_template('result.html', prediction=prediction)
            else:
                flash('Error predicting image')
                return redirect(request.url)
    return render_template('index.html')

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"500 error: {error}")
    return f"500 error: {error}", 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"An error occurred: {e}")
    return f"An error occurred: {e}", 500

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
