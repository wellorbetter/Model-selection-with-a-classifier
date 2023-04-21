import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from pyESN.pyESN import ESN
from sklearn.svm import SVR
from sklearn_extensions.extreme_learning_machines.elm import ELMRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from arch import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.svm import SVR
import torch.nn.functional as F
import heapq
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import xgboost as xgb
from skrebate import ReliefF
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
from sklearn import svm
import xgboost as xgb
# 加载数据
dataFileName = "F:\\Documents\\WeChat Files\\wxid_r7cfmedsjt8y22\\FileStorage\\File\\2023-04\\原始数据.xlsx"
data = pd.read_excel(dataFileName,
                     sheet_name="END")
# dataFileName = 'F:\\Documents\\JupyterNotebook\\data\\cude_oil_prediction\\week_data.csv'
# data = pd.read_csv(dataFileName)
data_names = ["CPI", "NHG", "CFNAI", "UER", "FPO", "ICO", "ECO", "USR", "NYF", "CUM", "UMS", "CUO", "NG", "DJI", "SP500",
              "DXY", "GOLD", "SV", "HIS", "US10B", "BP", "EC", "UC", "WTI"]
# data_names = ['DJI', 'Gold', 'Silver', 'Dollar', 'USD', 'WTI']
data_values = {}
for i in range(len(data_names)):
    data_values[data_names[i]] = data[data_names[i]].values

X, y = [], []
tmp_X, tmp_y = [], []

#  这个WTI和Brent不加
for i in range(len(data_names) - 1):
    if len(tmp_X) == 0:
        tmp_X = data_values[data_names[i]].reshape(-1, 1)
    else:
        tmp_X = np.concatenate((tmp_X, data_values[data_names[i]].reshape(-1, 1)), axis=1)

tmp_y = np.array(data_values["WTI"]).reshape(-1, 1)

scaler = StandardScaler()
tmp_X = scaler.fit_transform(tmp_X)
tmp_y = scaler.fit_transform(tmp_y)# 训练Lasso模型

lasso = Lasso(alpha=0.01)
lasso.fit(tmp_X, tmp_y)

# XGBoost 特征选择
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror')
xgb_model.fit(tmp_X, tmp_y)
importance = xgb_model.feature_importances_

# MI 特征选择
mi = mutual_info_regression(tmp_X, tmp_y)
# 打印每个特征的重要性得分
print('XGBoost feature importance: ', importance)
print('MI feature importance: ', mi)
# 输出选择的特征
selected_features = []
# for i in range(len(lasso.coef_)):
#     if abs(lasso.coef_[i]) > 0:
#         selected_features.append(i)

# 使用 heapq.nlargest() 函数找出最大的六个数和对应的下标
# 8
largest_six = heapq.nlargest(8, enumerate(mi), key=lambda x: x[1])
for i, v in largest_six:
    print(f"第{i}个数为{v}")
    selected_features.append(i)

print(lasso.coef_)
print(selected_features)
selected_features.append(-1)

for i in range(len(data_values["WTI"]) - 6):
    tmp = []
    for j in range(len(selected_features)):
        tmp.append(data_values[data_names[selected_features[j]]][i: i+6])
    X.append(tmp)
    y.append(data_values["WTI"][i + 6])

X = np.array(X)
y = np.array(y)

features_num = len(data_names) - 1
selected_features_num = len(selected_features)
X = X.reshape(len(X), -1)
print(X.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.298, random_state=42)
validate_size = 84

X_validate = X_test[:validate_size]
y_validate = y_test[:validate_size]
X_test = X_test[validate_size:]
y_test = y_test[validate_size:]
print(len(X_train))
print(len(X_validate))
print(len(X_test))