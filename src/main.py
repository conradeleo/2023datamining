from sklearn.preprocessing import LabelEncoder
import torch
from torch import optim
from torch.multiprocessing import Process, Queue
from tqdm import tqdm

from preprocess import DataProcessor
from model import GRU, LinearModel, SMAPELoss
from train import TrainEngine
import config

from argparse import ArgumentParser
import multiprocessing as mp 

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", default="train_2.csv")
parser.add_argument("-t", "--timestep", type=int, default=5)
parser.add_argument("-b", "--batch_size", type=int, default=64)

parser.add_argument("-e", "--epochs", type=int, default=10)
parser.add_argument("-npp", "--n_parallel_process", type=int, default=5)

args = parser.parse_args()

print('Preprocessing...')
dp = DataProcessor(args.dataset, args.timestep, args.batch_size)
train_loader, val_loader, test_loader = dp()

print('Model generating...')
GRU_T = GRU(2*args.timestep, 10, 1, 1)#定义GRU网络
ST_combined = LinearModel(config.feature_size+1, 1)
#train_loss_function = torch.nn.MSELoss()
train_loss_function = SMAPELoss()
test_loss_function = SMAPELoss()#定义损失函数
optimizer = optim.AdamW(GRU_T.parameters(), lr=config.learning_rate) # 定义优化器

#train_engine = TrainEngine(GRU_T, ST_combined, train_loss_function, test_loss_function, optimizer)

print('Training...')
#8.模型训练
for epoch in range(args.epochs):
    GRU_T.train()
    running_loss = 0
    train_bar = tqdm(train_loader, f"Train epoch {epoch+1}/{args.epochs}") #形成进度条
    for train_input, train_target, train_feature in train_bar:
        optimizer.zero_grad()

        output_gru = GRU_T(train_input)
        feature_expanded = train_feature.unsqueeze(1).expand(-1, output_gru.size(1), -1)
        combined_output = torch.cat([output_gru, feature_expanded], dim=2)
        final_output = ST_combined(combined_output)
        
        loss = train_loss_function(final_output, train_target)
        loss.backward()
        optimizer.step()
        
        train_bar.set_postfix(loss=loss.item())
        running_loss += loss.item()
        #train_bar.desc = "train epoch[()/()] loss:[:.3f]".format(epoch + 1, config.epochs, loss)#模型验证

    # Validation loss
    val_loss = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, f"Val epoch {epoch+1}/{args.epochs}") #形成进度条
        for val_input, val_target, val_feature in val_loader:
            output_gru = GRU_T(val_input)
            feature_expanded = val_feature.unsqueeze(1).expand(-1, output_gru.size(1), -1)
            combined_output = torch.cat([output_gru, feature_expanded], dim=2)
            final_output = ST_combined(combined_output)
        
            val_loss += train_loss_function(final_output, val_target)

    train_bar.set_postfix(loss=loss.item(), val_loss=val_loss.item())

GRU_T.eval()
test_loss = 0
with torch.no_grad():
    test_bar = tqdm(test_loader, f'Test')
    for test_input, test_target, test_feature in test_bar:
        output_gru = GRU_T(test_input)
        feature_expanded = test_feature.unsqueeze(1).expand(-1, output_gru.size(1), -1)
        combined_output = torch.cat([output_gru, feature_expanded], dim=2)
        final_output = ST_combined(combined_output)
        
        test_loss = test_loss_function(final_output, test_target)

    if test_loss < config.best_loss:
        config.best_loss = test_loss
        torch.save(GRU_T.state_dict(), config.save_path)

print(f'Test Loss: {test_loss.item():.4f}')

print('Finished Training')
'''
#9.绘制结果
plot_size = 200
plt.figure(figsize=(12,8))
plt.plot(scaler.inverse_transform((GRU_T(x_train_tesor).detach().numpy()[:plot_size]).reshape(-1,1)), "b")
plt.plot(scaler.inverse_transform(y_train_tesor.detach().numpy().reshape(-1,1)[:plot_size]), "r")
plt.legend()
plt.show

y_test_pred = GRU_T(x_test_tensor)
plt.figure(figsize=(12,8))
plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy()[:plot_size]), "b")
plt.plot(scaler.inverse_transform(y_test_tesor.detach().numpy().reshape(-1,1)[:plot_size]), "r")
plt.legend()
plt.show'''