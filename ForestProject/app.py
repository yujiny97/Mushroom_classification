from flask import Flask, render_template, request, jsonify
import model
from PIL import Image
import io
from keras.preprocessing import image
import multitest
import herb_multitest

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    return 'Hello Flask'

@app.route('/poisonresult', methods=['GET', 'POST'])
def poisionresult():
    try:
        if request.method=='POST':
            f = request.files['img'].read()
            f = Image.open(io.BytesIO(f))
            f = f.resize((64, 64))
            result = multitest.ispoison(f)
        elif request.method=='GET':
            print("GET")
            return jsonify({"status":"not OK", "message":"Not File"})
    except Exception:
        print("Exception")
        return jsonify({"status":"not OK", "message":"Not File"})
    return jsonify({"status":"OK","result":result})

@app.route('/herbresult', methods=['GET', 'POST'])
def herbresult():
    try:
        if request.method=='POST':
            f = request.files['img'].read()
            f = Image.open(io.BytesIO(f))
            f = f.resize((64, 64))
            result = herb_multitest.whatHerb(f)
        elif request.method=='GET':
            print("GET")
            return jsonify({"status":"not OK", "message":"Not File"})
    except Exception as e:
        print("Exception")
        print(e)
        return jsonify({"status":"not OK", "message":"Exception"})
    return jsonify({"status":"OK","result":result})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
