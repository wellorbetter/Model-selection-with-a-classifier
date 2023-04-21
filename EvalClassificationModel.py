from ClassificationModel import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
test_label = []
for i in range(len(X_test)):
    best_model = 0  # 最优模型的下标
    min_num = 999
    for j in range(len(model)):
        if mean_squared_error(
                scaler.inverse_transform(model[j].predict(X_test[i].reshape(1, len(X_test[i]))).reshape(-1, 1)),
                scaler.inverse_transform(y_test[i].reshape(-1, 1))).item() < min_num:
            min_num = mean_squared_error(
                scaler.inverse_transform(model[j].predict(X_test[i].reshape(1, len(X_test[i]))).reshape(-1, 1)),
                scaler.inverse_transform(y_test[i].reshape(-1, 1))).item()
            best_model = j
    test_label.append(best_model)

res = []
cnn_cnt = 0
# 下标放在后面展示看看
cnn_test_index = []
for i in range(len(X_test)):
    inputs = X_test[i]
    inputs = torch.from_numpy(inputs.astype(np.float32))
    choice = selector(inputs.reshape(1, 1, len(X_test[i])))
    cnn_test_index.append(torch.argmax(choice))
    predicted = model[int(torch.argmax(choice))].predict(X_test[i].reshape(1, len(X_test[i]))).item()
    res.append(predicted)
    if int(torch.argmax(choice)) == test_label[i]:
        cnn_cnt = cnn_cnt + 1
# 测试模型
myModel.eval()
print(X_test.shape)
with torch.no_grad():
    inputs = X_test
    inputs = torch.from_numpy(inputs.astype(np.float32))
    outputs = myModel(inputs)
    predicted = torch.argmax(outputs, dim=1)

for i in range(len(loss_list)):
    loss_list[i] = float(loss_list[i].detach())
plt.plot(range(len(loss_list[:500])), loss_list[:500], 'r')
plt.show()

print("Test Accuracy:")
cnt = 0
for i in range(len(X_test)):
    if predicted[i] == test_label[i]:
        cnt += 1
print("CNN", cnn_cnt / len(test_label))
print("MLP:", cnt / len(X_test))
#  svm分类器
svm_model = svm.SVC(kernel="linear", decision_function_shape="ovo")
svm_model.fit(X_validate, np.array(select_index).reshape(-1))
svm_test_predicted = svm_model.predict(X_test)
cnt = 0
for i in range(len(X_test)):
    if svm_test_predicted[i] == test_label[i]:
        cnt += 1
print("SVM:", cnt / len(X_test))
# 构建XGBoost模型
xgboost = xgb.XGBClassifier(
    max_depth=3,  # 决策树最大深度
    learning_rate=0.008,  # 学习率
    n_estimators=25,  # 决策树数量
    objective='multi:softmax',  # 指定多分类任务的目标函数
    num_class=len(model),  # 类别数
    booster='gbtree',  # 使用决策树作为基学习器
    n_jobs=4,  # 使用4个CPU核心并行计算
    seed=0  # 随机数种子
)

xgboost.fit(X_validate, select_index)
y_pred = xgboost.predict(X_test)
print("XGBoost:", sum(y_pred == test_label) / len(test_label))

# print("Test num")
# print("label", np.array(test_label).reshape(-1))
# print("CNN:", cnn_test_index)
# print("MLP:", predicted)
# print("SVM:", np.array(svm_test_predicted).reshape(-1))
# print("XGBoost:", np.array(y_pred).reshape(-1))

cnn_array = []
mlp_array = []
svm_array = []
for i in range(len(select_index)):
    # cnn
    inputs, labels = X_validate[i], select_index[i]
    inputs = torch.from_numpy(inputs.astype(np.float32))
    outputs = selector(inputs.reshape(1, 1, len(X_validate[i])))
    cnn_array.append(torch.argmax(outputs, dim=1))
    # mlp
    inputs = X_validate[i].reshape(1, len(X_validate[i]))
    labels = select_index[i]
    inputs = torch.from_numpy(inputs.astype(np.float32))
    labels = torch.from_numpy(np.array(labels).astype(np.float32))
    outputs = myModel(inputs)
    mlp_array.append(torch.argmax(outputs, dim=1))
    # svm
    svm_res = svm_model.predict(X_validate[i].reshape(1, len(X_validate[i])))
    svm_array.append(svm_res)

xgboost_array = xgboost.predict(X_validate)

# print("Validation num")
# print("validation_label:", select_index)
# print("CNN:", cnn_array)
# print("MLP:", mlp_array)
# print("SVM:", np.array(svm_array).reshape(-1))
# print("XGBoost:", xgboost_array)
cnn_cnt = 0
mlp_cnt = 0
svm_cnt = 0
for i in range(len(select_index)):
    if cnn_array[i].item() == select_index[i]:
        cnn_cnt += 1
    if mlp_array[i].item() == select_index[i]:
        mlp_cnt += 1
    if svm_array[i].item() == select_index[i]:
        svm_cnt += 1

print("validation set Accuracy:")
print("CNN:", cnn_cnt / len(select_index))
print("MLP:", mlp_cnt / len(select_index))
print("SVM:", svm_cnt / len(select_index))
print("XGBoost:", sum(xgboost.predict(X_validate) == select_index) / len(select_index))


