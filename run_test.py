import csv

import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model

# this already convert the date to numpy arrays easily
# variables:
#   passengerId                 (int)                                   0, 0
#   survived (train only)       (bool)                                  1, -
#   Pclass                      ('1', '2' or '3')                       2, 1
#   Name                        (str ----)                              3, 2
#   sex                         ('male' or 'female')                    4, 3
#   age                         (int of float, if < 1 or estimated)     5, 4
#   SibSp                       (int - siblings and spouses)            6, 5
#   Parch                       (int - parents of childs)               7, 6
#   Ticket                      (str ----)                              8, 7
#   Fare                        (float - price paid, I guess)           9, 8
#   Cabin                       (str ----)                             10, 9
#   Embarked                    (str: 'S', 'C' or 'Q')                 11,10

#### 0. Create dictionaries to map dataset
sex_map = {'male':0, 'female':1}
city_map = {'S':0, 'C':1, 'Q':2}
# features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
features = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare']

#### 1. load data in a numpy array
train_data = pd.read_csv('dataset/train.csv', quotechar='"')
test_data = pd.read_csv('dataset/test.csv', quotechar='"')
train_labels = train_data.columns.values.tolist()

#### 2. convert necessary things to ints and floats
train_data['Sex'] = train_data['Sex'].map(sex_map)
train_data['Embarked'] = train_data['Embarked'].map(city_map)

test_data['Sex'] = test_data['Sex'].map(sex_map)
test_data['Embarked'] = test_data['Embarked'].map(city_map)

#### 3. Build train and test array
X_train = train_data[features]
y_train = train_data['Survived']

X_test = test_data[features]

#### 4. Substitute NaNs for the features' means
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

#### X_train /= X_train.max()
#### X_test /= X_train.max()

#### 5. Apply logistic regression
readout = linear_model.LogisticRegression()
output_weights = readout.fit(X_train, y_train)
prediction = output_weights.predict(X_test)

#### 6. save results
survived = pd.DataFrame(prediction, columns=['Survived'])
predictionID = pd.DataFrame(test_data, columns=['PassengerId'])

results = pd.concat([predictionID, survived], axis=1)
results.to_csv('results.csv', index=False)
