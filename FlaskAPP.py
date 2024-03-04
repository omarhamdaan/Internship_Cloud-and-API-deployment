#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, jsonify
import numpy as np
import keras
import cv2

# App and model initializer
app = Flask(__name__)
title = 'Number Recognizer'

# Loading prebuilt AI
model = keras.models.load_model('mnist_classification.h5')

# GET method
@app.route('/')
def home():
    return render_template('home.html', title=title)
# POST method
@app.route('/', methods=['POST'])
def result():
    print('Post request received')
    file_str = request.files['file'].read()
    file_np = np.fromstring(file_str, np.uint8)
    print(f'File received: {file_np.shape}')

    # Decode the image
    img = cv2.imdecode(file_np, cv2.IMREAD_GRAYSCALE)

    # Resize the image to (28, 28)
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)

    # Normalize the pixel values to range [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0

    # Expand dimensions to match the model input shape
    img_processed = np.expand_dims(img_normalized, axis=0)
    img_processed = np.expand_dims(img_processed, axis=-1)  # Add channel dimension

    try:
        prediction = np.argmax(model.predict(img_processed))
        print(f"Prediction: {prediction}")
        response = jsonify(response=str(prediction), status=200)
    except Exception as e:
        response = jsonify(response=str(e), status=400)

    return response
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)  # Change the port number if needed

