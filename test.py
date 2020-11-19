
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from plotly import express as px
from plotly import figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score

import pandas as pd, numpy as np
import sys
import io

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.tree import export_graphviz
import pydot
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

# file = input("what is your file format? local CSV for 'L' ; sklearn_dataset for 'S' ; online_csv for 'O';,\n,\
# ohters for 'N'")
# if file == 'L' :
#     # local csv file
#     # lung.csv
#     file_name = input("what's your file name ?")
#     raw_data = pd.read_csv(file_name)
#     data = raw_data.to_numpy()
#     print('Successfully loaded')
# elif file == 'O' :
#     # online csv file
    # https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
# url = input('input URL')
# apt = pd.read_csv("https://raw.githubusercontent.com/alex31425/GEOG594-MTHuang/master/ApartmentF.csv")
# room = pd.read_csv("https://raw.githubusercontent.com/alex31425/GEOG594-MTHuang/master/RoomF.csv")
# data = raw_data.to_numpy()
# print(apt.columns)
# print(room.columns)

# for type in apt.dtypes:
#     if type == object:
#         print(type)

# objcolumn = apt.dtypes[apt.dtypes==object]
# print(objcolumn)
# objcolumn = list(objcolumn.index)
# print(objcolumn)
# apt = pd.get_dummies(apt,columns=objcolumn)

# print(apt.isnull().sum())
# apt = apt.dropna()
# print(apt.isnull().sum())
# print(room.dtypes)
# data = load_boston()
# df_apt = pd.DataFrame.from_records(apt.values)

# print(df_apt.dtypes)
# print(apt)
# X = df_apt.iloc[:,2:].to_numpy()
# y = df_apt.iloc[:,1].to_numpy()
# f = list(apt.columns)

# data = load_boston()
#
#
# X = data.data
# y = data.target
# f = data["feature_names"]
# print(X,'\n',y,'\n',f )

# elif file == 'S' :
#     data = load_boston()
#     print('Successfully loaded')
# else:
#     print('Other data type')
#
#
# try:
#     r = int(input('please enter the column number of response variable'))
#     y = data.T[r].T
#     print(y)
#     p1,p2 = [int(x) for x in input('please enter the column number of predictor variables').split()]
#     X = data.T[p1:p2].T
#     print(X)
# except:
#     print('value input error')

# data = load_boston()
# X = data.data
# y = data.target
# print(type(X), type(y))
# print(X,'\n',y)


# hist1_x = [0.8, 1.2, 0.2, 0.6, 1.6,
#            -0.9, -0.07, 1.95, 0.9, -0.2,
#            -0.5, 0.3, 0.4, -0.37, 0.6]
# hist2_x = [0.8, 1.5, 1.5, 0.6, 0.59,
#            1.0, 0.8, 1.7, 0.5, 0.8,
#            -0.3, 1.2, 0.56, 0.3, 2.2]
#
# hist_data = [hist1_x, hist2_x]
#
# group_labels = ['2012', '2013']
#
# rug_text_1 = ['a1', 'b1', 'c1', 'd1', 'e1',
#       'f1', 'g1', 'h1', 'i1', 'j1',
#       'k1', 'l1', 'm1', 'n1', 'o1']
#
# rug_text_2 = ['a2', 'b2', 'c2', 'd2', 'e2',
#       'f2', 'g2', 'h2', 'i2', 'j2',
#       'k2', 'l2', 'm2', 'n2', 'o2']
#
# rug_text_all = [rug_text_1, rug_text_2]
#
# fig = ff.create_distplot(
#     hist_data, group_labels, bin_size=.2)
# fig.show()

# import matplotlib.pyplot as plt
# x=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# y=[[1,2,3,4],[2,3,4,5],[3,4,5,6],[7,8,9,10]]
# colours=['r','g','b','k']
# plt.figure() # In this example, all the plots will be in one figure.
# for i in range(len(x)):
#     plt.plot(x[i],y[i],colours[i])
# plt.show()


df_pd =  pd.read_csv("https://raw.githubusercontent.com/IBM/ml-learning-path-assets/master/data/predict_home_value.csv")
print(df_pd.head())
categoricalColumns = df_pd.select_dtypes(include=[np.object]).columns

print("Categorical columns : " )
print(categoricalColumns)

impute_categorical = SimpleImputer(strategy="most_frequent")
onehot_categorical =  OneHotEncoder(handle_unknown='ignore')

categorical_transformer = Pipeline(steps=[('impute',impute_categorical),('onehot',onehot_categorical)])

# Defining the numerical columns
numericalColumns = [col for col in df_pd.select_dtypes(include=[np.int64]).columns if col not in ['SALEPRICE']]
print("Numerical columns : " )
print(numericalColumns)

scaler_numerical = StandardScaler()

numerical_transformer = Pipeline(steps=[('scale',scaler_numerical)])

preprocessorForCategoricalColumns = ColumnTransformer(transformers=[('cat', categorical_transformer, categoricalColumns)],
                                            remainder="passthrough")
preprocessorForAllColumns = ColumnTransformer(transformers=[('cat', categorical_transformer, categoricalColumns),('num',numerical_transformer,numericalColumns)],
                                            remainder="passthrough")


#. The transformation happens in the pipeline. Temporarily done here to show what intermediate value looks like
df_pd_temp = preprocessorForCategoricalColumns.fit_transform(df_pd)
print("Data after transforming :")
print(df_pd_temp)

df_pd_temp_2 = preprocessorForAllColumns.fit_transform(df_pd)
print("Data after transforming :")
print(df_pd_temp_2)
# prepare data frame for splitting data into train and test datasets

features = []
features = df_pd.drop(['SALEPRICE'], axis=1)

label = pd.DataFrame(df_pd, columns = ['SALEPRICE'])
#label_encoder = LabelEncoder()
label = df_pd['SALEPRICE']

#label = label_encoder.fit_transform(label)
print(" value of label : " + str(label))

X_train, X_test, y_train, y_test = train_test_split(features,df_pd['SALEPRICE'] , random_state=0)

print("Dimensions of datasets that will be used for training : Input features"+str(X_train.shape)+
      " Output label" + str(y_train.shape))
print("Dimensions of datasets that will be used for testing : Input features"+str(X_test.shape)+
      " Output label" + str(y_test.shape))

from sklearn.tree import DecisionTreeRegressor

model_name = "Decision Tree Regressor"

decisionTreeRegressor = DecisionTreeRegressor(random_state=0,max_features=30)

dtr_model = Pipeline(steps=[('preprocessorAll',preprocessorForAllColumns),('regressor', decisionTreeRegressor)])

dtr_model.fit(X_train,y_train)

y_pred_dtr = dtr_model.predict(X_test)

print(decisionTreeRegressor)

export_graphviz(decisionTreeRegressor, out_file ='tree.dot')
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')