from EvalClassificationModel import *
from statsmodels.tsa.arima.model import ARIMA
#  验证集上面的效果
cnn_predicted_value = []
mlp_predicted_value = []
svm_predicted_value = []
xgboost_predicted_value = []
for i in range(len(xgboost_array)):
    if model_name[xgboost_array[i]] == "ARIMA":
        xgboost_predicted_value.append(ARIMA(X_validate[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        xgboost_predicted_value.append(
            model[xgboost_array[i]].predict(X_validate[i].reshape(1, len(X_validate[i]))).item())
    if model_name[mlp_array[i]] == "ARIMA":
        mlp_predicted_value.append(ARIMA(X_validate[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        mlp_predicted_value.append(model[mlp_array[i]].predict(X_validate[i].reshape(1, len(X_validate[i]))).item())
    if model_name[cnn_array[i]] == "ARIMA":
        cnn_predicted_value.append(ARIMA(X_validate[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        cnn_predicted_value.append(model[cnn_array[i]].predict(X_validate[i].reshape(1, len(X_validate[i]))).item())
    if model_name[np.array(svm_array).reshape(-1)[i]] == "ARIMA":
        svm_predicted_value.append(ARIMA(X_validate[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        svm_predicted_value.append(
            model[np.array(svm_array).reshape(-1)[i]].predict(X_validate[i].reshape(1, len(X_validate[i]))).item())
print("Validation Set MSE:")
print("CNN:", mean_squared_error(scaler.inverse_transform(np.array(cnn_predicted_value).reshape(-1, 1)),
                                 scaler.inverse_transform(y_validate.reshape(-1, 1))))
print("MLP:", mean_squared_error(scaler.inverse_transform(np.array(mlp_predicted_value).reshape(-1, 1)),
                                 scaler.inverse_transform(y_validate.reshape(-1, 1))))
print("SVM:", mean_squared_error(scaler.inverse_transform(np.array(svm_predicted_value).reshape(-1, 1)),
                                 scaler.inverse_transform(y_validate.reshape(-1, 1))))
print("XGBoost:",
      mean_squared_error(scaler.inverse_transform(np.array(xgboost_predicted_value).reshape(-1, 1)),
                         scaler.inverse_transform(y_validate.reshape(-1, 1))))

print("Validation Set MAE:")
print("CNN:", mean_absolute_error(scaler.inverse_transform(np.array(cnn_predicted_value).reshape(-1, 1)),
                                  scaler.inverse_transform(y_validate.reshape(-1, 1))))
print("MLP:", mean_absolute_error(scaler.inverse_transform(np.array(mlp_predicted_value).reshape(-1, 1)),
                                  scaler.inverse_transform(y_validate.reshape(-1, 1))))
print("SVM:", mean_absolute_error(scaler.inverse_transform(np.array(svm_predicted_value).reshape(-1, 1)),
                                  scaler.inverse_transform(y_validate.reshape(-1, 1))))
print("XGBoost:",
      mean_absolute_error(scaler.inverse_transform(np.array(xgboost_predicted_value).reshape(-1, 1)),
                          scaler.inverse_transform(y_validate.reshape(-1, 1))))

print("Validation Set MAPE:")
print("CNN:", mean_absolute_percentage_error(scaler.inverse_transform(y_validate.reshape(-1, 1)),
      scaler.inverse_transform(np.array(cnn_predicted_value).reshape(-1, 1))))
print("MLP:", mean_absolute_percentage_error(scaler.inverse_transform(y_validate.reshape(-1, 1)),
      scaler.inverse_transform(np.array(mlp_predicted_value).reshape(-1, 1))))
print("SVM:", mean_absolute_percentage_error(scaler.inverse_transform(y_validate.reshape(-1, 1)),
      scaler.inverse_transform(np.array(svm_predicted_value).reshape(-1, 1))))
print("XGBoost:",
      mean_absolute_percentage_error(scaler.inverse_transform(y_validate.reshape(-1, 1)),
      scaler.inverse_transform(np.array(xgboost_predicted_value).reshape(-1, 1))))
#  测试集上面的效果
cnn_predicted_value = []
mlp_predicted_value = []
svm_predicted_value = []
xgboost_predicted_value = []
for i in range(len(test_label)):
    if model_name[y_pred[i]] == "ARIMA":
        xgboost_predicted_value.append(ARIMA(X_test[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        xgboost_predicted_value.append(model[y_pred[i]].predict(X_test[i].reshape(1, len(X_test[i]))).item())
    if model_name[predicted[i]] == "ARIMA":
        mlp_predicted_value.append(ARIMA(X_test[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        mlp_predicted_value.append(model[predicted[i]].predict(X_test[i].reshape(1, len(X_test[i]))).item())
    if model_name[cnn_test_index[i]] == "ARIMA":
        cnn_predicted_value.append(ARIMA(X_test[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        cnn_predicted_value.append(model[cnn_test_index[i]].predict(X_test[i].reshape(1, len(X_test[i]))).item())
    if model_name[np.array(svm_test_predicted).reshape(-1)[i]] == "ARIMA":
        svm_predicted_value.append(ARIMA(X_test[i], order=(2, 1, 2)).fit().forecast()[0])
    else:
        svm_predicted_value.append(
            model[np.array(svm_test_predicted).reshape(-1)[i]].predict(X_test[i].reshape(1, len(X_test[i]))).item())

print("Test Set MSE:")
print("CNN:", mean_squared_error(scaler.inverse_transform(np.array(cnn_predicted_value).reshape(-1, 1)),
                                 scaler.inverse_transform(y_test.reshape(-1, 1))))
print("MLP:", mean_squared_error(scaler.inverse_transform(np.array(mlp_predicted_value).reshape(-1, 1)),
                                 scaler.inverse_transform(y_test.reshape(-1, 1))))
print("SVM:", mean_squared_error(scaler.inverse_transform(np.array(svm_predicted_value).reshape(-1, 1)),
                                 scaler.inverse_transform(y_test.reshape(-1, 1))))
print("XGBoost:",
      mean_squared_error(scaler.inverse_transform(np.array(xgboost_predicted_value).reshape(-1, 1)),
                         scaler.inverse_transform(y_test.reshape(-1, 1))))

print("Test Set MAE:")
print("CNN:", mean_absolute_error(scaler.inverse_transform(np.array(cnn_predicted_value).reshape(-1, 1)),
                                  scaler.inverse_transform(y_test.reshape(-1, 1))))
print("MLP:", mean_absolute_error(scaler.inverse_transform(np.array(mlp_predicted_value).reshape(-1, 1)),
                                  scaler.inverse_transform(y_test.reshape(-1, 1))))
print("SVM:", mean_absolute_error(scaler.inverse_transform(np.array(svm_predicted_value).reshape(-1, 1)),
                                  scaler.inverse_transform(y_test.reshape(-1, 1))))
print("XGBoost:",
      mean_absolute_error(scaler.inverse_transform(np.array(xgboost_predicted_value).reshape(-1, 1)),
                          scaler.inverse_transform(y_test.reshape(-1, 1))))

print("Test Set MAPE:")
print("CNN:",
      mean_absolute_percentage_error(scaler.inverse_transform(y_test.reshape(-1, 1)),
      scaler.inverse_transform(np.array(cnn_predicted_value).reshape(-1, 1))))
print("MLP:",
      mean_absolute_percentage_error(scaler.inverse_transform(y_test.reshape(-1, 1)),
      scaler.inverse_transform(np.array(mlp_predicted_value).reshape(-1, 1))))
print("SVM:",
      mean_absolute_percentage_error(scaler.inverse_transform(y_test.reshape(-1, 1)),
      scaler.inverse_transform(np.array(svm_predicted_value).reshape(-1, 1))))
print("XGBoost:", mean_absolute_percentage_error(scaler.inverse_transform(y_test.reshape(-1, 1)),
      scaler.inverse_transform(np.array(xgboost_predicted_value).reshape(-1, 1))))
