import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

for i in range(50):
    #Had side effects, younger
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    #Didn't have side effects, older
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)
  
for i in range(950):
    #Didn't have side effects, younger
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    #Had side effects, older
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

for i in scaled_train_samples:
    print(i)

  

    