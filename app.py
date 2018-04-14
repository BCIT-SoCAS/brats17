from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug import secure_filename
from zipfile import ZipFile
from tumor_detector import TumorDetector
import os
import threading
from brain_img_processor import * 
from flask_compress import Compress
from test3d import *

status = "Free"
td = TumorDetector()
compress = Compress()

def start_app():
    app = Flask(__name__)
    compress.init_app(app)
    return app

app = start_app()
uploads_folder = os.path.join('data','tumor')
pred_model_path = os.path.join('pred_model', 'mri_modelv5.h5')
tumor_path = os.path.join('result', 'tumor.nii.gz')
  
@app.route('/', methods = ['GET'])   
def index():
    return render_template('index.html')  
 
@app.route('/brain_view', methods= ['GET'])
def show_main():

    flair_path = os.path.join(uploads_folder,'uploaded_MR_Flair','uploaded_MR_Flair.mha')
    t1_path = os.path.join(uploads_folder,'uploaded_MR_T1','uploaded_MR_T1.mha')
    t1c_path = os.path.join(uploads_folder,'uploaded_MR_T1c','uploaded_MR_T1c.mha')
    t2_path = os.path.join(uploads_folder,'uploaded_MR_T2','uploaded_MR_T2.mha')
    tumor_path = os.path.join('result','tumor.nii.gz')

    flair_data = BrainData(flair_path)
    t1_data = BrainData(t1_path)
    t1c_data = BrainData(t1c_path) 
    t2_data = BrainData(t2_path)
    tumor_data = BrainData(tumor_path) 
 
    brain = {
        'flair': flair_data.data.tolist(),
        't1': t1_data.data.tolist(),
        't1c': t1c_data.data.tolist(),
        't2': t2_data.data.tolist()
    }
    data = {
        'brain': brain,   
        'tumor': tumor_data.data.tolist() 
    }

    return render_template('main.html', data=data)
  
@app.route('/upload', methods = ['POST'])
def upload():

    status = "Free"
    f = request.files['file']
    rel_path = os.path.join(uploads_folder, secure_filename(f.filename))

    f.save(rel_path)

    # open and extract files to proper location
    with ZipFile(rel_path) as myzip:
        files = myzip.namelist()
        for file in files: 
            if file.endswith('Flair.mha'):
                bucket = 'uploaded_MR_Flair'
            elif file.endswith('T1.mha'):
                bucket = 'uploaded_MR_T1' 
            elif file.endswith('T2.mha'):
                bucket = 'uploaded_MR_T2' 
            elif file.endswith('T1c.mha'):
                bucket = 'uploaded_MR_T1c'

            path = os.path.join(uploads_folder, bucket)
            myzip.extract(file, path=path)
   
    # rename the files to proper names
    for root, dirs, files in os.walk(uploads_folder):
        for dir in dirs: 
            path = os.path.join(uploads_folder, dir)
            for root, dirs, files in os.walk(path):
                if len(files) > 0:
                    file = files[0]
                    file_path = os.path.join(path,file)
                    os.rename(file_path, os.path.join(path,'%s.mha' % dir))       
            
    # delete the zip file
    os.remove(rel_path) 

    thread = threading.Thread(target=run_test, args=(update_status,))
    thread.start()
    # call load_data from tumor_detector
    return render_template('processing.html') 
 
@app.route('/queryStatus', methods=['POST']) 
def query_status(): 
    return jsonify(status) 
 
@app.route('/predict', methods=['POST'])
def predict_survivability():
    enc_pred = getEncodedPrediction(pred_model_path,tumor_path)
    result = getLabelPrediction(enc_pred)
    return jsonify(result)

@app.route('/download', methods=['GET'])
def download():
    return send_from_directory(directory='result', filename='tumor.nii.gz', as_attachment=True)

def update_status(new_status):
    global status
    status = new_status

  
def run_test(cb): 
    cb('Configuring Neural Network') 
    td.start()
    cb('Loading Models')
    td.load_models()
    cb('Loading Data')
    td.load_data()
    cb('Testing Data on Models')
    td.start_test()
    cb('Done') 
    return

if __name__ == '__main__':
    
    app.run()
    