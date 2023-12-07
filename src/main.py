import torch
from torch import optim
from tqdm import tqdm

from preprocess import DataProcessor
from model import GRU, SMAPELoss

from argparse import ArgumentParser
import configparser

import os
import matplotlib.pyplot as plt
from datetime import datetime

def train_model(device, args, config):
    print('Preprocessing...')
    dp = DataProcessor(args.timestep, args.batch_size, args.train_mode)
    epoch_gen = dp(args.epochs)

    print('Model generating...')
    learning_rate = config['model_sett'].getfloat('learning_rate')
    feature_size = config['model_sett'].getint('feature_size')

    GRU_T = GRU(args.timestep, feature_size, 1, 1).to(device)
    loss_function = SMAPELoss().to(device)

    optimizer = optim.AdamW(GRU_T.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(5, args.epochs, 5)], gamma=0.7)
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

            output = GRU_T(train_input, train_feature)
            
            loss = loss_function(output, train_target)
            loss = loss.cpu()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_bar.set_postfix(loss=loss.item())
            running_loss.append(loss.item())

    GRU_T.eval()
    test_loss = 0
    plot_path = config['save_path']['plot_path']
    model_path = config['save_path']['model_path']
    best_loss = config['model_sett'].getfloat('best_loss')
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    with torch.no_grad():
        GRU_T.eval()
        test_bar = tqdm(test_loader, f'Test')
        for test_input, test_target, test_feature in test_bar:
            test_input = test_input.to(device)
            test_target = test_target.to(device)
            test_feature = test_feature.to(device)

            output = GRU_T(test_input, test_feature)
            
            test_loss = loss_function(output, test_target)
            test_loss = test_loss.cpu()

        if test_loss < best_loss:
            config['model_sett']['best_loss'] = str(test_loss.item())
            with open('config.ini', 'w') as configfile:
                config.write(configfile)

            torch.save(GRU_T.state_dict(), model_path)

    print(f'Test Loss: {test_loss.item():.4f}')
    
    plt.plot(running_loss, label='Training Loss')
    plt.xlabel('running')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_path}/{datetime.now().strftime("%Y%m%d_%H%M")}.png')
    print('Finished Training')

def test_result(device, args, config):
    print('Preprocessing...')
    dp = DataProcessor(args.timestep, args.batch_size, args.train_mode)
    epoch_gen = dp()

    print('Model loading...')
    feature_size = config['model_sett'].getint('feature_size')
    model_path = config['save_path']['model_path']
    GRU_T = GRU(args.timestep, feature_size, 1, 1).to(device)
    GRU_T.load_state_dict(torch.load(model_path))
    loss_function = SMAPELoss().to(device)

    print('Predicting...')
    final_loss = []
    for epoch, test_loader in epoch_gen:
        GRU_T.eval()
        test_bar = tqdm(test_loader, f'Test')
        for test_input, test_target, test_feature in test_bar:
            test_input = test_input.to(device)
            test_target = test_target.to(device)
            test_feature = test_feature.to(device)

            output = GRU_T(test_input, test_feature)
            
            test_loss = loss_function(output, test_target)
            final_loss.append(test_loss.cpu().item())

    final_score = sum(final_loss)/len(final_loss)
    print(f'FINAL SCORE IS: {final_score}!!!!!!')

def main():
    parser = ArgumentParser()
    parser.add_argument("--train_mode", action='store_true')
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--timestep", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)

    config = configparser.ConfigParser()
    config.read('config.ini')

    print('Start')
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available() & args.on_gpu
    print('Use gpu:', use_gpu)
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and args.on_gpu else "cpu")

    if (args.train_mode):
        print('Train model with train_1.csv')
        train_model(device, args, config)
    else:
        print('Test result with train_2,csv')
        test_result(device, args, config)

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