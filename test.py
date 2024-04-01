from pandas import read_csv
from bitarray import bitarray
from hdc_encoding import RecordEncoding
from binhdc import BinHDC
from sklearn.model_selection import train_test_split

# Load dataset
url = "dataset/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
features_minmax = {}
dataset = read_csv(url, names=names)
minGlobal = 1<<16
maxGlobal = 0
features_num = len(names) - 1
num_slices = 5
dimension = 100
num_classes = 3

for i in range(features_num): 
  minVal = min(dataset[names[i]])
  maxVal = max(dataset[names[i]])
  features_minmax[names[i]] = {'min': minVal, 'max': maxVal}

  if minVal < minGlobal:
    minGlobal = minVal

  if maxVal > maxGlobal:
    maxGlobal = maxVal

print(minGlobal, maxGlobal)

record_encoding = RecordEncoding(dimension, features_num, num_slices, minGlobal, maxGlobal)
hv_dataset = []

for index, row in dataset.iterrows():  
  hv = record_encoding.encode(row.values[0:-1])
  #print(hv)  
  hv_dataset.append(hv)


X = hv_dataset
y = dataset[names[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#print(hv_dataset)
BinHDC = BinHDC(num_classes, dimension, labels = classes)
BinHDC.fit(X_train, y_train)
y_pred = BinHDC.predict(X_test)
acc = BinHDC.accuracy(y_pred, y_test)
print(acc)
 

