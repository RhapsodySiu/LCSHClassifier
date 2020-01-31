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
from tensorflow import Graph, Session
from keras import backend as K
# feature extractor
from model.places365.vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
# subject heading classifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import pickle
#import tensorflow as tf
#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# MischaPanch
#from tensorflow.keras.backend import set_session
#from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
# quick fix on keras cannot load portrait classifier bc of GlorotUniform
from tensorflow.keras.models import load_model as tf_load_model

from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

test_img_path = './uploads/template.jpg'

IMAGE_DIMENSION = (224,224)
# MODEL_PATH = './model/keyword_places365_60subject_baseline.h5'
PORTRAIT_MODEL_PATH = './model/portrait_c_mobilenet_fold0.h5'
SUBJECT_MODEL_PATH = './model/sklearn_subject_places1365_extraTree_LinearSVC_baseline.sav'

CLASS_FILE = './label/subject60_top.json'
CLASS_LIST = {}
with open(CLASS_FILE, 'r') as json_file:
    CLASS_LIST = json.load(json_file)
    CLASS_LIST = CLASS_LIST['label']

# force tensorflow to use the global default graph
# by Satyajit
# global graph
# global sess
# sess = tf.Session()
# graph = tf.get_default_graph()
# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
# Otherwise, their weights will be unavailable in the threads after the session there has been set
# set_session(sess)
# load vgg16 base model
# this is temporary
# base_model = VGG16(weights=None, include_top=False)
# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(64,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# preds=Dense(len(CLASS_LIST), activation='softmax')(x) #final layer with softmax activation
# MODEL=Model(inputs=base_model.input,outputs=preds)
# MODEL.load_weights(MODEL_PATH)

# MODEL = load_model(MODEL_PATH)
GRAPH_PORTRAIT = Graph()
with GRAPH_PORTRAIT.as_default():
    SESSION_PORTRAIT = Session()
    with SESSION_PORTRAIT.as_default():
        PORTRAIT_MODEL = tf_load_model(PORTRAIT_MODEL_PATH)
#PORTRAIT_MODEL._make_predict_function()

SUBJECT_HEADING_MODEL = None
with open(SUBJECT_MODEL_PATH, 'rb') as f:
    SUBJECT_HEADING_MODEL = pickle.load(f)

# get vgg16 hybrid1365 model
GRAPH_EXTRACTION = Graph()
with GRAPH_EXTRACTION.as_default():
    SESSION_EXTRACTION = Session()
    with SESSION_EXTRACTION.as_default():
        EXTRACTION_MODEL = VGG16_Hybrid_1365(weights='places', include_top=False, input_shape=(224,224,3))
#EXTRACTION_MODEL._make_predict_function()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    subject_heading = 'error'
    is_portrait = 'Landscape'
    x = img_to_array(load_img(file, target_size=IMAGE_DIMENSION))
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    # try:
    #     #pred = MODEL.predict(x)
    #     #logging.debug(pred.shape)
    #     #pred_inc = np.argmax(pred[0])
    #     subject_heading = predict_subject_heading(x)
    #     logging.debug(subject_heading)
    #     is_portrait = predict_portrait(x)
    #     logging.debug(is_portrait)
    # except Exception as e:
    #     logging.debug(str(e))
    subject_heading = predict_subject_heading(x)
    # TODO: fix the portrait model loading problem
    #is_portrait = predict_portrait(x)
    return subject_heading, is_portrait

def predict_subject_heading(x):
    K.set_session(SESSION_EXTRACTION)
    with GRAPH_EXTRACTION.as_default():
        feature_vector = EXTRACTION_MODEL.predict(x)

    feature_vector = feature_vector.reshape((1, -1))
    pred = SUBJECT_HEADING_MODEL.predict(feature_vector)
    return pred[0]

def predict_portrait(x):
    K.set_session(SESSION_PORTRAIT)
    with GRAPH_PORTRAIT.as_default():
        pred = PORTRAIT_MODEL.predict(x)[0]
    pred_inc = np.argmax(pred)
    if pred_inc == 0:
        return 'Landscape'
    else:
        return 'Portrait'

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
            label, is_portrait = predict(file_path)
            logging.debug(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            logging.debug("--- %s seconds ---" % str (time.time() - start_time))
            #return render_template('template.html', label=label, isPortrait=is_portrait, imagesource='./uploads/' + filename)
            return jsonify(
                label=label,
                isPortrait=is_portrait,
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
