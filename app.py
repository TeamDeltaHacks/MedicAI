from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/author', methods=['GET'])
def author():
    return render_template('author.html')

@app.route('/create', methods=['GET'])
def create():
    return render_template('create.html')

@app.route('/details', methods=['GET'])
def details():
    return render_template('details.html')

@app.route('/explore', methods=['GET'])
def explore():
    return render_template('explore.html')