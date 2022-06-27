# MedicAI
Code for _**"Improving Clinical Efficiency and Reducing Medical Errors through NLP-enabled diagnosis of Health Conditions from Transcription Reports"**_
Top 3 @ STEMist education hacks
Pending Approval at Journal (6.14 Impact Factor)

**More Information:** <br>
[Research paper](https://github.com/CMEONE/MedicAI/blob/main/static/paper.pdf)<br>
[YouTube Demo](https://www.youtube.com/watch?v=_GsxYAZyjnU&feature=emb_title&ab_channel=KabirRamzan)

# Objective

In an effort to streamline the process for medical practitioners and improve clinical care for patients within the hospital, we developed various machine learning and deep learning based architectures to classify healthcare conditions in a transcription note, ultimately reducing misdiagnosis rates and alleviating the burden for overloaded hospitals. Furthermore, to increase the accessibility for hospitals world-wide, our models are packaged into an efficient and accurate AI-enabled medical app.

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
