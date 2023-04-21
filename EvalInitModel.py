
from TrainInitModel import *

all_mse = []
all_mae = []
all_mape = []
for i in range(len(model)):
    res = model[i].predict(X_validate)
    res_mse = mean_squared_error(scaler.inverse_transform(np.array(res).reshape(-1, 1)),
                                 scaler.inverse_transform(y_validate.reshape(-1, 1)))
    all_mse.append(res_mse)
    res_mae = mean_absolute_error(scaler.inverse_transform(np.array(res).reshape(-1, 1)),
                                  scaler.inverse_transform(y_validate.reshape(-1, 1)))
    all_mae.append(res_mae)
    res_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_validate.reshape(-1, 1)),
                                              scaler.inverse_transform(np.array(res).reshape(-1, 1)))
    all_mape.append(res_mape)
print("Mode Validation MSE")
print(all_mse)
print("Mode Validation MAE")
print(all_mae)
print("Mode Validation MAPE")
print(all_mape)

all_mse = []
all_mae = []
all_mape = []
for i in range(len(model)):
    res = model[i].predict(X_test)
    res_mse = mean_squared_error(scaler.inverse_transform(np.array(res).reshape(-1, 1)),
                                 scaler.inverse_transform(y_test.reshape(-1, 1)))
    all_mse.append(res_mse)
    res_mae = mean_absolute_error(scaler.inverse_transform(np.array(res).reshape(-1, 1)),
                                  scaler.inverse_transform(y_test.reshape(-1, 1)))
    all_mae.append(res_mae)
    res_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_test.reshape(-1, 1)),
                                              scaler.inverse_transform(np.array(res).reshape(-1, 1)))
    all_mape.append(res_mape)
print("Mode Test MSE")
print(all_mse)
print("Mode Test MAE")
print(all_mae)
print("Mode Test MAPE")
print(all_mape)