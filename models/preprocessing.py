import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

data = pd.read_csv('drive/MyDrive/June 24-25/mtsamples.csv') # change
data.loc[data.medical_specialty == ' Cardiovascular / Pulmonary', "medical_specialty"] = 'Heart'
data.loc[data.medical_specialty == ' Neurosurgery', 'medical_specialty'] = 'Brain'
data.loc[data.medical_specialty == ' Neurology', 'medical_specialty'] = 'Brain'
data.loc[data.medical_specialty == ' Urology', 'medical_specialty'] = 'Reproductive'
data.loc[data.medical_specialty == ' Obstetrics / Gynecology', 'medical_specialty'] = 'Reproductive'
data.loc[data.medical_specialty == ' Gastroenterology', 'medical_specialty'] = 'Digestive'
data.loc[data.medical_specialty == ' Nephrology', 'medical_specialty'] = 'Digestive'
data = data[data.medical_specialty.isin(['Heart', 'Brain', 'Reproductive', 'Digestive'])]
data['medical_specialty'].value_counts()

data = data[['transcription', 'medical_specialty']]
data = data.dropna()

data['medical_specialty'].value_counts()

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

data['transcription'] = data['transcription'].apply(clean)
data['transcription'] = data['transcription'].apply(lemmatize)