import tensorflow as tf
from tensorflow import keras 
from keras.models import Model
from keras.layers import Layer , Dense , Conv2D , MaxPooling2D , Flatten , Input
import numpy as np
from flask import Flask , request , jsonify , render_template
import os
import cv2

app = Flask(__name__)

def preprocess(file_path):
    byte_image = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_image)
    img = tf.image.resize(img , (100 , 100))
    img = img /255.0
    return img 

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
       
    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)


model = tf.keras.models.load_model('Facial_Recognition_Model.h5',
                                          custom_objects={'L1Dist': L1Dist,
                                                        'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

def verify(model, detection_threshold, verification_threshold):
    results = []
    
    verification_images = [f for f in os.listdir(os.path.join('Application_Data', 'Verification_images')) 
                         if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image in verification_images:
        validation_img = preprocess(os.path.join('Application_Data', 'Verification_images', image))
        input_img = preprocess(os.path.join('Application_Data', 'Input_image', 'Input_image.jpg'))
        
        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)
    
    detection = np.sum(np.array(results) > detection_threshold)
    verification = detection / len(verification_images)  
    verified = verification > verification_threshold
    
    return results, verified

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/capture')
def capture():
    return render_template('capture.html')

@app.route('/verify', methods=['POST'])
def verify_image():
    print("Verify endpoint called")  # Debug print
    if model is None:
        print("Model not loaded")  # Debug print
        return jsonify({"error": "Model not loaded", "verified": False})
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")  # Debug print
            return jsonify({"error": "Cannot open camera", "verified": False})
            
        ret, frame = cap.read()
        
        if ret:
            
            height, width = frame.shape[:2]
            start_y = (height - 250) // 2
            start_x = (width - 250) // 2
            frame = frame[start_y:start_y+250, start_x:start_x+250, :]
            
            input_path = os.path.join('Application_Data', 'Input_image', 'Input_image.jpg')
            cv2.imwrite(input_path, frame)
            print(f"Image saved to {input_path}")  # Debug print
            
            results, verified = verify(model, 0.5, 0.5)
            print(f"Verification result: {verified}")  # Debug print
            
            return jsonify({"verified": bool(verified)})
        
        print("Failed to capture image")  # Debug print
        return jsonify({"error": "Failed to capture image", "verified": False})
    
    except Exception as e:
        print(f"Error during verification: {str(e)}")  # Debug print
        return jsonify({"error": str(e), "verified": False})
    
    finally:
        cap.release()

if __name__ == '__main__':
    app.run(debug=True)
