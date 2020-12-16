import os
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split


data_path = './data/rri.csv'

# load csv data
df = pd.read_csv(data_path)
df = df.drop([0])

# 
rri_night = df['am4-3'].dropna().values.astype(float)
rri_noon = df['pm4-5'].dropna().values.astype(float)

plt.figure(figsize=(6,4))
plt.subplot(1, 1, 1)
plt.plot(rri_night)
plt.plot(rri_noon)
plt.legend(['night', 'noon'])
plt.ylim(600, 1400)
plt.title('RRI')
plt.show()

features = np.concatenate([rri_night, rri_noon])[:, np.newaxis]
targets = np.concatenate([np.zeros(rri_night.shape[0]), np.ones(rri_noon.shape[0])])

# split data into train set and test set
train_features, test_features, train_targets, test_targets = train_test_split(
    features, 
    targets, 
    test_size=0.2, 
    random_state=42 
)

# normalize features
sc = StandardScaler()
sc.fit(train_features)
normalized_train_features = sc.transform(train_features)
normalized_test_features = sc.transform(test_features)

# init svc classifier
svc_classifier = svm.SVC()

# train svc classifier
svc_classifier.fit(normalized_train_features, train_targets)

# predict using trained svc classifier
test_predictions = svc_classifier.predict(normalized_test_features)

# caluculate accuracy
accuracy = accuracy_score(test_targets, test_predictions)

print('accuracy -> {}%'.format(accuracy*100))