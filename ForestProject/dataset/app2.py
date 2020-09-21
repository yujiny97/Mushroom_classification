from flask import Flask, render_template, request, jsonify
import model
from PIL import Image
import io
from keras.preprocessing import image
import multitest

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
    return 'Hello Flask'

@app.route('/poisonresult', methods=['GET', 'POST'])
def poisionresult():
    f = request.files['img'].read()
    f = Image.open(io.BytesIO(f))
    f = f.resize((64, 64))
    result = multitest.ispoison(f)
    return jsonify({"status":"OK","result":result})



if __name__ == '__main__':
    app.run(host='0.0.0.0')