from sklearn.preprocessing import LabelEncoder
import torch
from torch import optim
from torch.multiprocessing import Process, Queue
from tqdm import tqdm

from preprocess import DataProcessor
from model import GRU, LinearModel, SMAPELoss
from train import TrainEngine

from argparse import ArgumentParser
import configparser

import os
import matplotlib.pyplot as plt
from datetime import datetime

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_mode", action='store_true')
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--timestep", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument('--loss_type', type=str, default="smape", choices=['smape', 'mse'])
    #parser.add_argument("--n_parallel_process", type=int, default=5)

    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Start')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available() & args.on_gpu
    print('Use gpu:', use_gpu)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and args.on_gpu else "cpu")

    print('Preprocessing...')
    dp = DataProcessor(args.timestep, args.batch_size, args.train_mode)
    epoch_gen = dp(args.epochs)

    print('Model generating...')
    learning_rate = config['model_sett'].getfloat('learning_rate')
    feature_size = config['model_sett'].getint('feature_size')

    GRU_T = GRU(args.timestep, 10, 1, 1).to(device)
    ST_combined = LinearModel(feature_size, 10, 1).to(device)
    train_loss_function = SMAPELoss().to(device) if args.loss_type == 'smape' else torch.nn.MSELoss().to(device)
    test_loss_function = SMAPELoss().to(device)

    optimizer = optim.AdamW(GRU_T.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(5, args.epochs, 5)], gamma=0.85)
    #train_engine = TrainEngine(GRU_T, ST_combined, train_loss_function, test_loss_function, optimizer)

    print('Training...')
    running_loss = []
    #8.模型训练
    for epoch, train_loader, test_loader in epoch_gen:
        GRU_T.train()
        train_bar = tqdm(train_loader, f"Train epoch {epoch+1}/{args.epochs}") #形成进度条
        for train_input, train_target, train_feature in train_bar:
            train_input = train_input.to(device)
            train_target = train_target.to(device)
            train_feature = train_feature.to(device)
            optimizer.zero_grad()

            output_gru = GRU_T(train_input)
            combined_output = torch.cat([output_gru, train_feature], dim=-1)
            final_output = ST_combined(combined_output)
            
            loss = train_loss_function(final_output, train_target)
            loss = loss.cpu()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_bar.set_postfix(loss=loss.item())
            running_loss.append(loss.item())
            #train_bar.desc = "train epoch[()/()] loss:[:.3f]".format(epoch + 1, config.epochs, loss)#模型验证

    GRU_T.eval()
    test_loss = 0
    plot_path = config['save_path']['plot_path']
    model_path = config['save_path']['model_path']
    best_loss = config['model_sett'].getfloat('best_loss')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with torch.no_grad():
        test_bar = tqdm(test_loader, f'Test')
        for test_input, test_target, test_feature in test_bar:
            test_input = test_input.to(device)
            test_target = test_target.to(device)
            test_feature = test_feature.to(device)

            output_gru = GRU_T(test_input)
            combined_output = torch.cat([output_gru, test_feature], dim=-1)
            final_output = ST_combined(combined_output)
            
            test_loss = test_loss_function(final_output, test_target)
            test_loss = test_loss.cpu()

        if test_loss < best_loss:
            config['model_sett']['best_loss'] = str(test_loss.item())
            with open('config.ini', 'w') as configfile:
                config.write(configfile)

            torch.save(GRU_T, f'{model_path}/GRU')
            torch.save(ST_combined, f'{model_path}/ST_Linear')

    print(f'Test Loss: {test_loss.item():.4f}')
    
    plt.plot(running_loss, label='Training Loss')
    plt.xlabel('running')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_path}/{datetime.now().strftime("%Y%m%d_%H%M")}_{args.loss_type}.png')
    #plt.show()
    print('Finished Training')

if __name__ == '__main__':
    main()
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