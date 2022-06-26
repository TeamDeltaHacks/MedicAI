from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import pytesseract
import numpy as np
import urllib
import json as JSON
from joblib import dump, load
import re

import nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

model = load("LR.pkl")

def clean(text):
	text = text.lower()
	text = text.strip()
	text = re.compile('[/(){}\[\]\|@,;]').sub(' ', text) 
	text = re.compile('[^0-9a-z #+_]').sub('', text) 
	words = text.split()
	i = 0 
	while i < len(words):
		if words[i] in stopwords.words('english'):
			words.pop(i)
		else:
			i += 1
	
	return ' '.join(map(str, words))

def lemmatize(text):
	wordlist=[]
	lemmatizer = WordNetLemmatizer() 
	sentences=sent_tokenize(text)
	
	for sentence in sentences:
		words=word_tokenize(sentence)
		for word in words:
			wordlist.append(lemmatizer.lemmatize(word))	
	return ' '.join(wordlist)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
	return render_template('index.html')

@app.route('/records', methods=['GET'])
def view_records():
	return render_template('records.html')

@app.route('/records/<record>', methods=['GET'])
def records(record):
	return render_template('record.html', record=record)

@app.route('/add', methods=['GET', 'POST'])
def add():
	if(request.method == 'GET'):
		return render_template('add.html')
	else:
		try:
			json = request.json
			if("report" not in json or json["report"] == ""):
				return JSON.dumps({
					"type": "error",
					"response": "Missing field: image."
				})

			req = urllib.request.urlopen(json["report"])
			arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
			image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
			custom_config = r'--oem 3 --psm 6'
			parsed = pytesseract.image_to_string(image, config=custom_config)

			paragraphs = parsed.split("\n\n")
			longest_paragraph = paragraphs[0]

			for i in range(1, len(paragraphs)):
				if(len(longest_paragraph) < len(paragraphs[i])):
					longest_paragraph = paragraphs[i]

			longest_paragraph = longest_paragraph.replace("\n", " ")

			cleaned = clean(longest_paragraph)
			print(cleaned)

			lemmatized = lemmatize(cleaned)
			print(lemmatized)

			return JSON.dumps({
				"type": "success",
				"response": {
					"parsed": parsed,
					"longest": longest_paragraph
				}
			})
		except Exception as e:
			print(e)
			return JSON.dumps({
				"type": "error",
				"response": "Invalid request, please try again."
			})

@app.route('/register', methods=['GET'])
def register():
	return render_template('register.html')

@app.route('/login', methods=['GET'])
def login():
	return render_template('login.html')