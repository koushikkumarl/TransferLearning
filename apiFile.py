import numpy as np 
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
from werkzeug.utils import secure_filename

from flask import Flask, redirect, url_for, request, render_template 
 
apiFile = Flask(__name__) 
 
 
model_path = 'vgg19.h5'

model = load_model(model_path) 
 
model.make_predict_function() 
 

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(244, 244))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x) 
    
    preds = model.predict(x)
    return preds


@apiFile.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@apiFile.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        preds = model_predict(file_path, model)
        pred_class = decode_predictions(preds, top=1)
        result = str(pred_class[0][0][1])
        return result 
    return None
        


        
if __name__ == '__main__':
    apiFile.run(debug=True)