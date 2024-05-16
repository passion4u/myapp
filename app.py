from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model('banknote_classifier.h5')
categories = ['Japan_banknote', 'USA_banknote', 'Euro_banknote', 'UK_banknote', 'China_banknote', 'Canada_banknote', 'Australia_banknote', 'India_banknote', 'Brazil_banknote', 'Russia_banknote']

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (50, 50))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return categories[np.argmax(prediction)]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        file_path = f'uploads/{file.filename}'
        file.save(file_path)
        prediction = predict_image(file_path)
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
