# coding: utf-8

# # Astroturfing Review System

# Created By : Eashan Adhikarla (aeashan@gmail.com)
# Date : November'2015

# Basically I am importing the datasets and storing the datas in three columns : 
# *Polarity of the review
# *Review itself
# *True or Deceptive as ('t' or 'd')
# 
# Approach:
#a.) Import the datasets and store the data in three columns: Polarity of the review | Review itself | True or Deceptive as ('t' or 'd').
#b.) Converting 't' to 1 and 'd' to 0 because I will be using this as my target value and the review as my feature.
#c.) Splitting the Review data into testing data and training data (0.3 and 0.7 respectively).
#d.) Using CountVectorizer() to extract numeric features of each of the review as classifier can only use numeric data to compute something.
#e.) Using Multinomial Naive Bias method classifier to classify the reviews as Deceptive/True.
#

import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split


# **Setting up the folder paths in which the dataset is presetn**
# 

pos_deceptive_folder_path = 'op_spam_v1.4/positive_polarity/deceptive_from_MTurk/'
pos_true_folder_path = 'op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/'

neg_deceptive_folder_path = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk/'
neg_true_folder_path = 'op_spam_v1.4/negative_polarity/truthful_from_Web/'


# **Initialising the lists in which the polarity, review and either it's fake or true will be stored**


polarity_class = []
reviews = []
spamity_class =[]


# ** Since we have 5 folders in each folder in our dataset, I am using a for loop to iterate through each of the folder and collect datas (i.e Polarity, Review, Fake or True) and store**


for i in range(1,6):
    
    insidendec = neg_deceptive_folder_path + 'fold' + str(i)  # negative and fake
    insidentru = neg_true_folder_path + 'fold' + str(i) # negative and true

    insidepdec = pos_deceptive_folder_path + 'fold' + str(i) # positive and fake
    insideptru = pos_true_folder_path + 'fold' + str(i) # positive and true
    
    pos_list = [] # positive list ?
    
    for data_file in sorted(os.listdir(insidendec)): # negative and fake
        polarity_class.append('negative')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidendec, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insidentru)): # negative and true
        polarity_class.append('negative')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidentru, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insidepdec)): # positive and fake
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidepdec, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insideptru)): # positive and true
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insideptru, data_file)) as f:
                contents = f.read()
                reviews.append(contents)


# ** Making the dataframe using pandas to store polarity, reviews and true or fake **

# *Setting '0' for deceptive review and '1' for true review*


data_fm = pd.DataFrame({'polarity_class':polarity_class,'review':reviews,'spamity_class':spamity_class})

data_fm.loc[data_fm['spamity_class']=='d','spamity_class']=0
data_fm.loc[data_fm['spamity_class']=='t','spamity_class']=1


# ** Splitting the dataset to training and testing (0.7 and 0.3)**


data_x = data_fm['review']

data_y = np.asarray(data_fm['spamity_class'],dtype=int)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_x, data_y, test_size=0.3)


# ** Using CountVectorizer() method to extract features from the text reviews and convert it to numeric data, which can be used to train the classifier **

# *Using fit_transform() for X_train and only using transform() for X_test*

cv =  CountVectorizer()

X_traincv = cv.fit_transform(X_train)
X_testcv = cv.transform(X_test)


# **Using Naive Bayes Multinomial method as the classifier and training the data**

nbayes = MultinomialNB()

nbayes.fit(X_traincv, y_train)


# **Predicting the fake or deceptive reviews**
# 

# *using X_testcv : which is vectorized such that the dimensions are matched*

y_predictions = nbayes.predict(X_testcv)


# ** Printing out fake or deceptive reviews **

y_result = list(y_predictions)
yp=["True" if a==1 else "Deceptive" for a in y_result]
X_testlist = list(X_test)
output_fm = pd.DataFrame({'Review':X_testlist ,'True(1)/Deceptive(0)':yp})
output_fm


# ** Printing out the Accuracy, Precision Score, Recall Score, F1 Score **


print("Accuracy % :",metrics.accuracy_score(y_test, y_predictions)*100)
print("Precision Score: ", precision_score(y_test, y_predictions, average='micro'))
print("Recall Score: ",recall_score(y_test, y_predictions, average='micro') )
print("F1 Score: ",f1_score(y_test, y_predictions, average='micro') )
