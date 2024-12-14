import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import matplotlib.pyplot as plt


# 1. 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = output_dim

        # 1.1 输入嵌入层
        self.embedding = nn.Linear(input_dim, d_model)

        # 1.2 位置编码层
        self.pos_encoder = PositionalEncoding(d_model)

        # 1.3 Transformer 编码器层
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout),
            num_layers=num_layers
        )

        # 1.4 输出层
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # 2.1 输入嵌入
        src = self.embedding(src)  # (seq_len, batch_size, input_dim) -> (seq_len, batch_size, d_model)

        # 2.2 位置编码
        src = self.pos_encoder(src)

        # 2.3 通过编码器
        output = self.encoder_layers(src)  # (seq_len, batch_size, d_model) -> (seq_len, batch_size, d_model)

        # 2.4 输出
        output = self.fc(output[-1])  # 取最后一个时间步输出 (seq_len, batch_size, d_model) -> (batch_size, output_dim)
        return output


# 2. 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return x


# 3. 训练数据准备（生成一些示例数据）
def generate_data(seq_len, num_samples):
    time = np.arange(0, seq_len * num_samples).reshape((num_samples, seq_len))
    data = np.sin(time / 20) * np.cos(time / 5)
    data = data.astype(np.float32)
    return torch.tensor(data), torch.tensor(data[:, -1])  # 返回数据和最后一个时间步的值


# 4. 超参数
input_dim = 1
d_model = 32
nhead = 4
num_layers = 2
output_dim = 1
seq_len = 100
num_samples = 500
batch_size = 32
learning_rate = 0.001
num_epochs = 50

# 5. 模型，优化器，损失函数
model = Transformer(input_dim, d_model, nhead, num_layers, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 6. 训练
train_data, train_labels = generate_data(seq_len, num_samples)

for epoch in range(num_epochs):
    epoch_loss = 0
    for i in range(0, num_samples, batch_size):
        batch_data = train_data[i:i + batch_size].unsqueeze(-1).permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        batch_labels = train_labels[i:i + batch_size].unsqueeze(-1)  # (batch_size, output_dim)

        optimizer.zero_grad()
        outputs = model(batch_data)  # (batch_size, output_dim)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch: {epoch + 1}, Loss: {epoch_loss / (num_samples / batch_size):.4f}')

# 7. 测试
test_data, test_labels = generate_data(seq_len, 100)
model.eval()

with torch.no_grad():
    predictions = model(test_data.unsqueeze(-1).permute(1, 0, 2)).squeeze()

# 8. 可视化
plt.plot(test_labels.numpy(), label='Actual')
plt.plot(predictions.numpy(), label='Predicted')
plt.legend()
plt.show()