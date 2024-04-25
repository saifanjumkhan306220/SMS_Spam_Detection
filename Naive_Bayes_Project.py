#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
df=pd.read_csv(r'D:\FINAL SEMESTER\encoded_spam.csv')
df
# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
df.head()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')
import string
import nltk
from nltk.corpus import stopwords
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)
df['transformed_text'] = df['text'].apply(transform_text)
df.head()
X = tfidf.fit_transform(df['transformed_text']).toarray()
X.shape
y = df['target'].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print("Accuracy - ", accuracy_score(y_test,y_pred1))
print("Confusion Matrix - \n", confusion_matrix(y_test,y_pred1))
print("Precision - ",precision_score(y_test,y_pred1))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predictions using Gaussian Naive Bayes model
y_pred_gnb = gnb.predict(X_test)

# Calculate confusion matrix for Gaussian Naive Bayes
cm_gnb = confusion_matrix(y_test, y_pred_gnb)

# Plotting confusion matrix for Gaussian Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', cbar=False)  # Corrected variable name
plt.title('Confusion Matrix - Gaussian Naive Bayes')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
import matplotlib.pyplot as plt
# Predictions using Gaussian Naive Bayes model
y_pred_gnb = gnb.predict(X_test)
# Accuracy scores
# accuracy_scores = [accuracy_score(y_test, y_pred1), accuracy_score(y_test, y_pred2), accuracy_score(y_test, y_pred3)]
# Calculate accuracy, confusion matrix, and precision score for Gaussian Naive Bayes model
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
confusion_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
precision_gnb = precision_score(y_test, y_pred_gnb)
# Plotting accuracy and precision for Gaussian Naive Bayes model
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(['Naive Bayes'], [accuracy_gnb], color='skyblue')
plt.title('Accuracy Scores')
plt.xlabel('Model')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.bar(['Naive Bayes'], [precision_gnb], color='salmon')
plt.title('Precision Scores')
plt.xlabel('Model')
plt.ylabel('Precision')

# Show plot
plt.tight_layout()
plt.show()
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[ ]:




