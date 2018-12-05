

```python
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
```


```python
# **Setting up the folder paths in which the dataset is presetn**
# 
pos_deceptive_folder_path = 'op_spam_v1.4/positive_polarity/deceptive_from_MTurk/'
pos_true_folder_path = 'op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/'

neg_deceptive_folder_path = 'op_spam_v1.4/negative_polarity/deceptive_from_MTurk/'
neg_true_folder_path = 'op_spam_v1.4/negative_polarity/truthful_from_Web/'

```


```python
# **Initialising the lists in which the polarity, review and either it's fake or true will be stored**
polarity_class = []
reviews = []
spamity_class =[]
```


```python
print(polarity_class)
```

    []



```python
# ** Since we have 5 folders in each folder in our dataset, 
# I am using a for loop to iterate through each of the folder and collect datas 
# (i.e Polarity, Review, Fake or True) and store**

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
```


```python
print(polarity_class.count("negative"))
```

    800



```python
print(polarity_class.count("positive"))
```

    800



```python
# ** Making the dataframe using pandas to store polarity, reviews and true or fake **
# *Setting '0' for deceptive review and '1' for true review*

data_fm = pd.DataFrame({'polarity_class':polarity_class,'review':reviews,'spamity_class':spamity_class})
```


```python
data_fm.loc[data_fm['spamity_class']=='d','spamity_class']=0
```


```python
data_fm.loc[data_fm['spamity_class']=='t','spamity_class']=1
```


```python
# ** Splitting the dataset to training and testing (0.7 and 0.3)**
data_x = data_fm['review']
```


```python
data_y = np.asarray(data_fm['spamity_class'],dtype=int)
```


```python
print(data_y)
```

    [0 0 0 ..., 1 1 1]



```python
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data_x, data_y, test_size=0.3)
```


```python
# ** Using CountVectorizer() method to extract features from the text reviews and convert it to numeric data, which can be used to train the classifier **

# *Using fit_transform() for X_train and only using transform() for X_test*

cv =  CountVectorizer()
```


```python
X_traincv = cv.fit_transform(X_train)
```


```python
X_testcv = cv.transform(X_test)
```


```python
# **Using Naive Bayes Multinomial method as the classifier and training the data**
nbayes = MultinomialNB()
```


```python
nbayes.fit(X_traincv, y_train)
```




    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)




```python
# **Predicting the fake or deceptive reviews**
# 

# *using X_testcv : which is vectorized such that the dimensions are matched*
y_predictions = nbayes.predict(X_testcv)
```


```python
# ** Printing out fake or deceptive reviews **
y_result = list(y_predictions)
yp = ["True" if a==1 else "Deceptive" for a in y_result]
X_testlist = list(X_test)
output_fm = pd.DataFrame({'Review':X_testlist ,'True(1)/Deceptive(0)':yp})
output_fm
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Review</th>
      <th>True(1)/Deceptive(0)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>My sister and I stayed on the 22nd floor, ther...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Thank god I got this hotel through priceline. ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>This was a great hotel right in the heart of t...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>This has been one of the most pleasurable hote...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Our stay at the Ambassador East Hotel was extr...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The James Chicago is a stuffy, uninviting hote...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I would definitely recommend this hotel to any...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Perfect location, clean and courteous staff al...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>8</th>
      <td>I was REALLY looking forward to a nice relaxin...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>9</th>
      <td>During my recent trip to Chicago, I stayed at ...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>10</th>
      <td>I booked my weekend Chicago stay at the Hard R...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>11</th>
      <td>The Omni Chicago Hotel offers all the great am...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>12</th>
      <td>I truly enjoyed my stay at the Omni Chicago Ho...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>13</th>
      <td>I really enjoyed staying at this hotel. The se...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>14</th>
      <td>took a weekend trip with my wife. got a great ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>our stay was absolutely perfect. its a cool ho...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>16</th>
      <td>The Good LOCATION Views Decor Room Service Fri...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Having come to Chicago with my three sisters f...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>18</th>
      <td>I have stayed at many hotels traveling for bot...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>19</th>
      <td>If you are looking for a luxurious downtown Ch...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>20</th>
      <td>My husband and I stayed at the Sheraton Chicag...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>We booked 2 rooms for 2 nights on Hotwire over...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>The elevator system was impossible. It seems t...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>The Omni Chicago is hands down, the best hotel...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>24</th>
      <td>As a frequent traveler for work, I stay at man...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>I recently stayed at the Hyatt Regency in Chic...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>26</th>
      <td>My husband and I snagged a great deal on a wee...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>27</th>
      <td>The hotel is located in an hard to fins locati...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sofitel Chicago Water Tower is a four star hot...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>29</th>
      <td>My boyfriend and I were amazed by the breathta...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>930</th>
      <td>My stay at the Hyatt Regency was an experience...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>931</th>
      <td>This hotel was PERFECT for our girls getaway. ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>932</th>
      <td>For the price, you would think this would be a...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>933</th>
      <td>I just spent a week at the Millennium Knickerb...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>934</th>
      <td>My husband and I satayed for two nights at the...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>935</th>
      <td>I found this wonderful Hotel ! The location is...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>936</th>
      <td>This hotel has a good name and a good location...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>937</th>
      <td>Located right in the heart of downtown Chicago...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>938</th>
      <td>Fairmont Chicago was a great choice for my wif...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>939</th>
      <td>Terrible experience, I will not stay here agai...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>940</th>
      <td>We stayed here because of the enthusiastic pos...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>941</th>
      <td>at check in, the girl gave us a smoking room. ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>942</th>
      <td>I stayed at the Hilton Chicago for my cousins ...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>943</th>
      <td>My husband and I arrived for a 3 night stay fo...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>944</th>
      <td>My wife and I's stay at the Sheraton Chicago ...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>945</th>
      <td>My family of four went to a convention and sta...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>946</th>
      <td>I got married in the Chicago area this passed ...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>947</th>
      <td>IMO the Amalfi doesnt come close to justifying...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>948</th>
      <td>great expectations from the hotel of THE FUGIT...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>949</th>
      <td>The Sofitel Chicago Water Tower, while seeming...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>950</th>
      <td>I recently stayed at the Affinia Hotel in Chic...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>951</th>
      <td>First off, don't get a room on a lower floor, ...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>952</th>
      <td>My husband and I were really looking forward t...</td>
      <td>Deceptive</td>
    </tr>
    <tr>
      <th>953</th>
      <td>We stay at Hilton for 4 nights last march. It ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>954</th>
      <td>i valeted my lexus and it was returned with a ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>955</th>
      <td>This hotel was chosen by my husband's company ...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>956</th>
      <td>The Hilton Chicago is one of the best Hotels I...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>957</th>
      <td>We ate at the Prime House restaurant last year...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>958</th>
      <td>What's not to like? --&gt; Problems getting the r...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>959</th>
      <td>The Sheraton is a fantastic hotel. My wife and...</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>960 rows × 2 columns</p>
</div>




```python
# ** Printing out the Accuracy, Precision Score, Recall Score, F1 Score **

print("Accuracy % :",metrics.accuracy_score(y_test, y_predictions)*100)
print("Precision Score: ", precision_score(y_test, y_predictions, average='binary'))
print("Recall Score: ",recall_score(y_test, y_predictions, average='binary') )
print("F1 Score: ",f1_score(y_test, y_predictions, average='binary') )
```

    Accuracy % : 93.2291666667
    Precision Score:  0.953091684435
    Recall Score:  0.912244897959
    F1 Score:  0.932221063608



```python

```
