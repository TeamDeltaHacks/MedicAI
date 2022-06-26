# MedicAI

Code for "Improving Clinical Efficiency and Reducing Medical Errors through NLP-enabled diagnosis of Health Conditions from Transcription Reports"

GitHub repository for [research paper](https://github.com/CMEONE/MedicAI/blob/main/static/paper.pdf) and [web application](https://devpost.com/software/medicai-r2hsvu).

## Dependencies
You may need to use `pip install` to install the dependencies on each line of `requirements.txt` (for example, `pip install flask==1.1.2`). This list is not comprehensive and does not include many of the modules, which will need to be installed as errors arise with running (see below section).

You will also need to install Tesseract, more details can be found online or at the [Tesseract User Manual](https://github.com/tesseract-ocr/tessdoc).

## Running
To run the code, open a terminal and navigate to the root directory of this repository. Then, run the following commands:
```bash
export FLASK_APP=app.py
flask run -h localhost -p 5001
```

As import errors arise, you may need to install more modules by using `pip install MODULE_NAME_HERE`.

Finally, once the program runs without errors, navigate to `localhost:5001` in your browser. Enjoy!
