
from flask import Flask, request, render_template, redirect
import joblib
import os
import numpy as np
import tensorflow, keras
from keras.preprocessing import image
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input


app = Flask(__name__)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model1 =joblib.load('/home/janani/rubberproj/defectdetection/models/rproj_fnl_okay_random_forest_classifier.pkl')
model2 = joblib.load('/home/janani/rubberproj/defectdetection/models/rproj_fnl_notokay_random_forest_classifier.pkl')

img_size = 224
class_names_first_model = ['okay', 'notokay']
class_names_second_model = ['cut', 'flash']

def extract_features_from_image(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = base_model.predict(img_array).flatten()
    return features

def predict_image_class(model, image_path, img_size, class_names):
    img_features = extract_features_from_image(image_path, img_size)
    img_features = img_features.reshape(1, -1)
    prediction = model.predict(img_features)
    predicted_label = class_names[int(prediction[0])]
    prediction_scores = model.predict_proba(img_features)[0]
    return predicted_label, prediction_scores

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            upload_folder = 'defectdetection/static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
                
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            
            predicted_label, predictions = predict_image_class(model1, file_path, img_size, class_names_first_model)
            
            if predicted_label == 'notokay':
                predicted_label2, predictions2 = predict_image_class(model2, file_path, img_size, class_names_second_model)
                return render_template('result.html', 
                                       img_path='uploads/' + file.filename, 
                                       predicted_label=predicted_label,
                                       predictions=predictions,
                                       predicted_label2=predicted_label2,
                                       predictions2=predictions2)
            return render_template('result.html', 
                                   img_path='uploads/' + file.filename, 
                                   predicted_label=predicted_label,
                                   predictions=predictions)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
