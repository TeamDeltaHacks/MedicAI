from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import io
import base64

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/records', methods=['GET'])
def view_records():
	return render_template('records.html')

@app.route('/records/<record>', methods=['GET'])
def records():
	return render_template('record.html', record=record)

@app.route('/add', methods=['GET', 'POST'])
def add():
	if(request.method == 'GET'):
		return render_template('add.html')
	else:
		json = request.json
		return '{"type":"success","response":"result"}'

@app.route('/register', methods=['GET'])
def register():
	return render_template('register.html')

@app.route('/login', methods=['GET'])
def login():
	return render_template('login.html')