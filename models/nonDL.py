import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.metrics import classification_report

data = pd.read_csv('drive/MyDrive/June 24-25/mtsamples.csv')
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

data.rename(columns = {'transcription':'Report', 'medical_specialty':'speciality'}, inplace = True)

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

vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',ngram_range=(1,3), max_df=0.75, use_idf=True, smooth_idf=True, max_features=1000)
tfIdfMat  = vectorizer.fit_transform(data['Report'].tolist() )
feature_names = sorted(vectorizer.get_feature_names())
print(feature_names)

import gc
gc.collect()
pca = PCA(n_components=0.95)
tfIdfMat_reduced = pca.fit_transform(tfIdfMat.toarray())
labels = data['speciality'].tolist()
category_list = data.speciality.unique()
X_train, X_test, y_train, y_test = train_test_split(tfIdfMat_reduced, labels, stratify=labels, test_size=0.2, random_state=42)

clf = LogisticRegression().fit(X_train, y_train)
clf = RandomForestClassifier().fit(X_train, y_train)
y_test_pred= clf.predict(X_test)

print(classification_report(y_test,y_test_pred,labels=category_list))

accuracy_score(y_test, y_test_pred)

import gc
gc.collect()
tfIdfMatrix = tfIdfMat.todense()
labels = data['speciality'].tolist()
tsne_results = TSNE(n_components=2,init='random',random_state=0, perplexity=40).fit_transform(tfIdfMatrix)
plt.figure(figsize=(16,10))
palette = sns.hls_palette(4, l=.6, s=.9)
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels,
    palette= palette,
    legend="full",
    alpha=0.3
)
plt.show()

import joblib
joblib.dump(clf, 'LR.pkl')