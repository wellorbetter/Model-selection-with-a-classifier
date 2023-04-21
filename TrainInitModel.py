from LoadData import *


# 训练模型，注意这个X的维度，在输入lstm的时候应该有所不同
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train.reshape(-1))

# 训练MLP模型并预测
mlp_model = MLPRegressor(hidden_layer_sizes=(80,), activation='relu', solver='adam', max_iter=600, random_state=42)

mlp_model.fit(X_train, y_train)

esn_model = ESN(n_inputs=selected_features_num * 6, n_outputs=1, n_reservoir=2500, spectral_radius=0.2, random_state=42)
esn_model.fit(X_train, y_train)

ls_svr_model = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1)
ls_svr_model.fit(X_train, y_train.reshape(-1).ravel())

#  RVM的训练
rvm = BayesianRidge(n_iter=300, verbose=True)
rvm.fit(X_train, y_train)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_size = input_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(selected_features_num * hidden_size, 1)

    def forward(self, x):
        x = self.lstm(x)
        x = self.fc(x[0].reshape(len(x[0]), x[0].shape[1] * x[0].shape[2]))
        return x

    def predict(self, x):
        return self.forward(torch.from_numpy(x).type(torch.Tensor).view(len(x), selected_features_num, 6)).detach().reshape(-1)


lstm = LSTM(6, 30)
loss = nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-2)
max_epochs = 100
lstm_train_data = torch.from_numpy(X_train).type(torch.Tensor).view(len(X_train), selected_features_num, 6)
lstm_train_label = torch.from_numpy(y_train).type(torch.Tensor).view(-1)
for epoch in range(max_epochs):
    pre = lstm(lstm_train_data)
    pre_loss = loss(pre.view(-1), lstm_train_label)
    pre_loss.backward()
    optimizer.step()
    optimizer.zero_grad()


# 构建ELM模型并进行训练
elm = ELMRegressor(n_hidden=60, activation_func='sigmoid', random_state=42, alpha=0.2)
elm.fit(X_train, y_train)

model = [mlp_model, esn_model, ls_svr_model, lstm, rvm, rf]
model_name = ["MLP", "ESN", "LS_SVM", "LSTM", "RVM", "RandomForest"]
