# util
import os
import json
import uuid
import time
import logging

import logging
logging.basicConfig(level=logging.DEBUG)

# web app
from flask import Flask, render_template, session, redirect, url_for, request, jsonify
import requests
from werkzeug import secure_filename

# prediction
import numpy as np

# subject heading classifier
import pickle
import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.models import load_model

# configure tensorflow 
tf_config = tf.ConfigProto(log_device_placement=False, device_count={'GPU':1})
sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()

test_img_path = './uploads/test.jpg'

IMAGE_DIMENSION = (224,224)
PORTRAIT_MODEL_PATH = './model/portrait_mobilenet'
KEYWORD_MODEL_PATH = './model/keyword_vgg16'
DISTRICT_MODEL_PATH = './model/district_vgg16'
HKLCSH60_MODEL_PATH = './model/HKLCSHext60_std_logisticOvR.pkl'
HKLCSH275_MODEL_PATH = './model/HKLCSH275_logisticOvR.pkl'
HKLCSH570_MODEL_PATH = './model/HKLCSH570_logisticOvR.pkl'
SCALER60_PATH = './model/resnet_scaler.pkl'
HKLCSH60_LABEL_PATH = './label/HKLCSH60_labels.json'
HKLCSH275_LABEL_PATH = './label/HKLCSH275_labels.json'
HKLCSH570_LABEL_PATH = './label/HKLCSH570_labels.json' 

# load labels 
with open(HKLCSH60_LABEL_PATH, 'r') as f:
    HKLCSH60_LABELS = list(json.load(f).keys())
with open(HKLCSH275_LABEL_PATH, 'r') as f:
    HKLCSH275_LABELS = list(json.load(f).keys())
with open(HKLCSH570_LABEL_PATH, 'r') as f:
    HKLCSH570_LABELS = list(json.load(f).keys())
with open(SCALER60_PATH, 'rb') as f:
    scaler = pickle.load(f)
    
# get extractor, stage1 models
set_session(sess)
EXTRACTOR = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
PORTRAIT_CLF = load_model(PORTRAIT_MODEL_PATH)
KEYWORD_CLF = load_model(KEYWORD_MODEL_PATH)
DISTRICT_CLF = load_model(DISTRICT_MODEL_PATH)

# load logistic models 
with open(HKLCSH60_MODEL_PATH, 'rb') as f:
    HKLCSH60_MODEL = pickle.load(f)
with open(HKLCSH275_MODEL_PATH, 'rb') as f:
    HKLCSH275_MODEL = pickle.load(f)
with open(HKLCSH570_MODEL_PATH, 'rb') as f:
    HKLCSH570_MODEL = pickle.load(f)

PORTRAIT_LABELS = ['Landscape', 'Portrait']
KEYWORD_LABELS = ['Aerial Photos', 'Airportss', 'Ancestral Halls', 'Bank Buildings', 'Beaches', 'Cemeteries', 'Cinemas', 'Commercial Buildings', 'Costumes', 'Festivals', 'Government Buildings/Offices', 'Horse Racing', 'Hotels', 'Housing', 'Markets', 'Museums', 'Night Views', 'Parks', 'Performing Venues', 'Piers', 'Religious Buildings', 'Reservoirs', 'Restaurants', 'Schools', 'Typhoon Shelters']
DISTRICT_LABELS = ['Central and Western District', 'Eastern District', 'Islands District', 'Kowloon City District', 'Kwai Tsing District', 'Kwun Tong District', 'North District', 'Sai Kung District', 'Sha Tin District', 'Sham Shui Po District', 'Southern District', 'Tai Po District', 'Tsuen Wan District', 'Tuen Mun District', 'Wan Chai District', 'Wong Tai Sin District', 'Yau Tsim Mong District', 'Yuen Long District']

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def get_input(img_path):
    img = load_img(img_path, target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x 
    
def get_feature(x, extractor):
    global graph
    with graph.as_default():
        set_session(sess)
        feature = extractor.predict(x)
        return feature
    
def predict_stage1(x):
    global graph 
    with graph.as_default():
        set_session(sess)
        y_pred = PORTRAIT_CLF.predict(x)
        portrait = "{} {:.2f}%".format(PORTRAIT_LABELS[np.argmax(y_pred[0])], np.max(y_pred[0])*100) 
        y_pred = DISTRICT_CLF.predict(x)
        district = "{} {:.2f}%".format(DISTRICT_LABELS[np.argmax(y_pred[0])], np.max(y_pred[0])*100)
        y_pred = KEYWORD_CLF.predict(x)
        keyword = "{} {:.2f}%".format(KEYWORD_LABELS[np.argmax(y_pred[0])], np.max(y_pred[0])*100)
        return (portrait, district, keyword)

def parse_prediction(y_score, label_list, top_n=3):
    '''
    Parse to score to prediction
    y_score: [n*l] 2d array where n is sample size and l is number of labels
    label: list of labels. each index corresponds to that label name
    top_n: result to return for each sample
    multilabel: boolean. Whether the y_score is multilabel or not.
    
    return: [length = sample size] array of dict{'label': [top_n label], 'proba': [top_n probability]}
    '''
    result = []
    label_list = np.asarray(label_list)
    for y in y_score:
        indices = y.argsort()[-top_n:][::-1]
        proba = y[indices]
        label = label_list[indices]
        result.append({'label': label, 'proba': proba})
    return result
    
def predict(file):
    try:
        x = get_input(file)
        portrait, district, keyword = predict_stage1(x)
        feature = get_feature(x, EXTRACTOR)
        feature = scaler.transform(feature)
        # HKLCSH60/275/570
        y_score = HKLCSH60_MODEL.predict_proba(feature)
        result60 = parse_prediction(y_score, HKLCSH60_LABELS)
        y_score = HKLCSH275_MODEL.predict_proba(feature)
        result275 = parse_prediction(y_score, HKLCSH275_LABELS)
        y_score = HKLCSH570_MODEL.predict_proba(feature)
        result570 = parse_prediction(y_score, HKLCSH570_LABELS)
        result60_str = ""
        for label, prob in zip(result60[0]['label'], result60[0]['proba']):
            result60_str += "| {} {:.2f}% ".format(label, prob*100)
        result275_str = ""
        for label, prob in zip(result275[0]['label'], result275[0]['proba']):
            result275_str += "| {} {:.2f}% ".format(label, prob*100)
        result570_str = ""
        for label, prob in zip(result570[0]['label'], result570[0]['proba']):
            result570_str += "| {} {:.2f}% ".format(label, prob*100)
        result = {'subject60': result60_str, 'subject275': result275_str, 'subject570': result570_str, 'is_portrait': portrait, 'keyword': keyword, 'district': district}
        return result
    except Exception as e:
        error_msg = "Error in backend"
        result = {'subject60': error_msg, 'subject275': error_msg, 'subject570': error_msg, 'is_portrait': error_msg, 'keyword': error_msg, 'district': error_msg}
        return result 

# from ferrygun/AIFlowers2
def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

# from ferrygun/AIFlowers2
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_main():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # get portrait prediction 
            result = predict(file_path)
            logging.debug(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            logging.debug("--- %s seconds ---" % str (time.time() - start_time))
            #return render_template('template.html', label=label, isPortrait=is_portrait, imagesource='./uploads/' + filename)
            return jsonify(
                subject60=result['subject60'],
                subject275=result['subject275'],
                subject570=result['subject570'],
                isPortrait=result['is_portrait'],
                district=result['district'],
                keyword=result['keyword'],
                imagesource='./uploads/' + filename
            )

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.debug=True
    app.run(host='0.0.0.0', port=3000)
