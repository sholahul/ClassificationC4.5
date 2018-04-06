import numpy as np #scientific computing package
import graphviz
import pandas as pd #dataframe,series
import seaborn as sns
import pydotplus
import io

from matplotlib import pyplot as plt, path
from sklearn.model_selection import KFold # import KFold
from sklearn import tree
from matplotlib import pyplot as plt
from scipy import misc
from sklearn.tree import DecisionTreeClassifier , export_graphviz


#prosedur
def getData(pathToData):
    dataPre = []
    with open(pathToData) as f:
        dataPre = f.readlines()
    dataPra = []
    for d in dataPre:
        dataPra.append([x for x in d.split(',')])
    data = np.array(dataPra)
    return data

def splitData(data):
    attribute = np.array(data[0])
    X = np.array(data[1:,:6])
    Y = np.array(data[1:,6])

    kf = KFold(n_splits=3)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
    return attribute, x_train, x_test, y_train, y_test

def show_tree(tree,features,path):
    f = io.StringIO()
    export_graphviz(tree,out_file=f ,feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = misc.imread(path)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img)


#main
tdata = pd.read_csv('car.csv')
tdata.describe()
tdata.info()
tdata.head( )
data = getData('car.csv') #import data csv
attribute, x_train, x_test, y_train, y_test = splitData(data) #get data from csv in prosedur split data

print("========== Praproses Data ============")
print("Jumlah data : ",len(data)-1)
print("Jumlah data training : ",len(x_train))
print("Jumlah data testing : ",len(x_test))

print(data)

print("Data training")
print(x_train,y_train)

print("Data Testing")
print(x_test,y_test)

print("\n\n=========== Implement C4.5 ========== ")

dt = c.fit(x_train,y_train)
show_tree(dt,features,'dec_tree_01.png')
