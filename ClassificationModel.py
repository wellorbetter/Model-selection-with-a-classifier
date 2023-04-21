from EvalInitModel import *
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 输入词向量是10维
        # 这里自己设计一下，先就用一层的卷积
        # 输入 (N, C_in, L_in) 输出 (N, C_out, L_out) 这个channel，我这里是 (1, 1, 6)
        self.linear1 = nn.Linear(selected_features_num * 6, 101)  # 这里变成了 1 1 101 方便后面多层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=80, kernel_size=2)
        # 这里变成了 1 80 100
        self.norm1 = nn.BatchNorm1d(80)  # 归一化了
        self.maxpool1 = nn.MaxPool1d(2, 2)
        # 1 80 50
        self.conv2 = nn.Conv1d(in_channels=80, out_channels=40, kernel_size=2)  # 1 40 49
        self.norm2 = nn.BatchNorm1d(40)
        self.maxpool2 = nn.MaxPool1d(2, 2)  # 1 40 24

        self.conv3 = nn.Conv1d(in_channels=40, out_channels=20, kernel_size=2)  # 这里是 1 20 23
        self.norm3 = nn.BatchNorm1d(20)
        self.maxpool3 = nn.MaxPool1d(2, 2)  # 1 20 11

        self.conv4 = nn.Conv1d(in_channels=20, out_channels=10, kernel_size=2)  # 这里是 1 10 10
        self.norm4 = nn.BatchNorm1d(10)
        self.maxpool4 = nn.MaxPool1d(2, 2)  # 1 10 5

        self.fc = nn.Linear(50, len(model))  # 开始做的回归，效果不好，整个分类看看
        self.lrelu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.lrelu(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.lrelu(x)
        x = self.maxpool4(x)
        x = F.dropout(x, p=0.2)
        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)
        x = x + 0.001 * l2_reg
        x = self.fc(x.reshape(-1))
        x = self.softmax(x.reshape(1, -1))
        return x

select_index = []
for i in range(len(X_validate)):
    best_model = 0  # 最优模型的下标
    min_num = 999
    for j in range(len(model)):
        if mean_squared_error(
                scaler.inverse_transform(
                    model[j].predict(X_validate[i].reshape(1, len(X_validate[i]))).reshape(-1, 1)),
                scaler.inverse_transform(y_validate[i].reshape(-1, 1))).item() < min_num:
            min_num = mean_squared_error(
                scaler.inverse_transform(
                    model[j].predict(X_validate[i].reshape(1, len(X_validate[i]))).reshape(-1, 1)),
                scaler.inverse_transform(y_validate[i].reshape(-1, 1))).item()
            best_model = j
    select_index.append(best_model)
print(select_index)

# 拿到了之后要训练

cnn_pre_index = []  # cnn 预测的下标
selector = CNN()
epochs = 20
train_loss_all = []
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(selector.parameters(), lr=0.001)
for epoch in range(epochs):
    train_correct = 0
    for i in range(len(select_index)):
        inputs, labels = X_validate[i], select_index[i]
        inputs = torch.from_numpy(inputs.astype(np.float32))
        outputs = selector(inputs.reshape(1, 1, len(X_validate[i])))
        optimizer.zero_grad()
        loss_epoch = loss(outputs, torch.Tensor(np.array([labels])).long())
        loss_epoch.backward()
        optimizer.step()
        cnn_pre_index.append(int(torch.argmax(outputs, dim=1)))
        train_loss_all.append(loss_epoch)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(selected_features_num * 6, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, len(model))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x


# 实例化模型
myModel = MyModel()

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)

# 训练模型
epochs = 20
loss_list = []
for epoch in range(epochs):
    # 前向传播
    for i in range(len(select_index)):
        inputs = X_validate[i].reshape(1, len(X_validate[i]))
        labels = select_index[i]
        inputs = torch.from_numpy(inputs.astype(np.float32))
        labels = torch.from_numpy(np.array(labels).astype(np.float32))
        outputs = myModel(inputs)
        # print(outputs.float())
        # print(torch.Tensor([labels]).long())
        loss = criterion(outputs, torch.Tensor([labels]).long())
        # 反向传播和优化
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
