'''The following module trains the weights of the neural network model.'''
import sys

sys.path.append("./")
import os
import random
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from data_declaration import Task
from loader_helper import LoaderHelper

# Network structures
from networks.conv_Densenet121 import densenet121
from networks.ssm_nnMamba import nnMambaEncoder
from networks.tr_ViT import ViT
from networks.tr_crate import CRATE_small_3D

from evaluation import evaluate_model
from torchmetrics import AUROC, Accuracy, Specificity, Recall

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 改為使用 GPU 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")

if not torch.cuda.is_available():
    print("⚠️  警告: 未檢測到 GPU，某些模型（如 nnMamba）需要 CUDA 支援")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_weights(model_in, uuid_arg, epoch, fold=1, task: Task = None):
    """Save model weights with timestamp"""
    root_path = f"../weights/{task}/{uuid_arg}/"
    os.makedirs(root_path, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_path = f"{root_path}fold_{fold}_epoch{epoch}_weights-{timestamp}.pth"
    
    while os.path.exists(save_path):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") 
        save_path = f"{root_path}fold_{fold}_epoch{epoch}_weights-{timestamp}.pth"
        
    torch.save(model_in.state_dict(), save_path)

def save_best_weights(model_in, uuid_arg, task: Task):
    """Save model weights to task-specific folder"""
    root_path = f"../weights/{task}/{uuid_arg}/"
    os.makedirs(root_path, exist_ok=True)
    torch.save(model_in.state_dict(), f"{root_path}best_weight.pth")

def load_model(pth):
    '''Function for loaded camull net from a specified weights path'''
    model = torch.load(pth)
    print("Model loaded from weights file.", pth)
    model.to(DEVICE)
    return model


def build_arch(name):
    '''Function for instantiating the pytorch neural network object'''
    if name == 'densenet':
        net = densenet121(mode="classifier", drop_rate=0.05, num_classes=2)
    elif name == 'vit':
        net = ViT(in_channels=1, img_size=(112, 136, 112), patch_size=(8, 8, 8), pos_embed='conv', classification=True)
    elif name == 'nnmamba':
        net = nnMambaEncoder()
    elif name == 'crate':
        # net = CRATE_base_3D()
        net = CRATE_small_3D()
        # net = CRATE_tiny_3D()

    net.to(DEVICE)
    net.float()
    return net

def evaluate(model_in, test_dl, thresh=0.5, param_count=False):
    model_in.eval()
    
    # Initialize metrics
    auroc = AUROC(task='binary').to(DEVICE)
    accuracy_metric = Accuracy(task='binary', threshold=thresh).to(DEVICE)
    specificity_metric = Specificity(task='binary', threshold=thresh).to(DEVICE)
    sensitivity_metric = Recall(task='binary', threshold=thresh).to(DEVICE)  # Recall is same as Sensitivity
    
    total_label = torch.tensor([]).to(DEVICE)
    total_pred = torch.tensor([]).to(DEVICE)
    
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_dl):
            batch_X = sample_batched['mri'].to(DEVICE)
            batch_y = sample_batched['label'].to(DEVICE)
            
            net_out = model_in(batch_X)
            net_out = net_out.sigmoid().detach()
            
            total_label = torch.cat((total_label, batch_y), 1)
            total_pred = torch.cat((total_pred, net_out), 1)
    
    # Calculate metrics
    auc = auroc(total_pred, total_label).item()
    accuracy = accuracy_metric(total_pred, total_label).item()
    specificity = specificity_metric(total_pred, total_label).item()
    sensitivity = sensitivity_metric(total_pred, total_label).item()
    
    # Round the results
    accuracy = round(accuracy, 5)
    sensitivity = round(sensitivity, 5)
    specificity = round(specificity, 5)
    auc = round(auc, 5)
    
    # 清理 GPU 記憶體
    del total_label, total_pred, auroc, accuracy_metric, specificity_metric, sensitivity_metric
    torch.cuda.empty_cache()
    
    return accuracy, sensitivity, specificity, auc

def train_loop(model_in, train_dl, test_dl, epochs, uuid_, k_folds, task):
    '''Function containing the neural net model training loop'''
    optimizer = optim.AdamW(model_in.parameters(), lr=0.0001, weight_decay=1e-3)
    # optimizer = optim.AdamW(model_in.parameters())
    # scheduler_warm = lr_scheduler.ConstantLR(optimizer, start_factor=0.2, total_iters=5)
    scheduler_warm = lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)
    # loss_function = nn.BCELoss()
    loss_function = nn.BCEWithLogitsLoss()
    # loss_function = nn.CrossEntropyLoss()
    loss_fig = []
    eva_fig = []
    model_in.train()
    best_auc = 0
    nb_batch = len(train_dl)

    log_path = "../train_log/" + uuid_ + ".txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path):
        filein = open(log_path, 'a')
    else:
        filein = open(log_path, 'w')

    # Train
    for i in range(1, epochs + 1):
        loss = 0.0
        model_in.train()
        for _, sample_batched in enumerate(tqdm(train_dl)):
            batch_x = sample_batched['mri'].to(DEVICE)

            # batch_clinical = sample_batched['clin_t'].to(DEVICE)
            batch_y = sample_batched['label'].to(DEVICE)

            model_in.zero_grad()
            # outputs = model_in(batch_x,batch_clinical)
            outputs = model_in(batch_x)
            batch_loss = loss_function(outputs, batch_y)
            batch_loss.backward()
            optimizer.step()
            loss += float(batch_loss) / nb_batch

        tqdm.write("Epoch: {}/{}, train loss: {}".format(i, epochs, round(loss, 5)))
        filein.write("Epoch: {}/{}, train loss: {}\n".format(i, epochs, round(loss, 5)))
        loss_fig.append(round(loss, 5))

        # 每 5 個 epoch 才評估一次，提升訓練效率
        if i % 5 == 0 or i == epochs:
            # 測試集評估
            test_acc, test_sens, test_spec, test_auc = evaluate(model_in, test_dl)
            eva_fig.append(test_acc)
            
            # 訓練集評估（檢查過擬合）
            train_acc, train_sens, train_spec, train_auc = evaluate(model_in, train_dl)
            
            tqdm.write("Epoch: {}/{}, Test: Acc={}, AUC={} | Train: Acc={}, AUC={}".format(
                i, epochs, test_acc, test_auc, train_acc, train_auc))
            filein.write("Epoch: {}/{}, Test: Acc={}, Sens={}, Spec={}, AUC={} | Train: Acc={}, AUC={}\n".format(
                i, epochs, test_acc, test_sens, test_spec, test_auc, train_acc, train_auc))
            
            # 檢查過擬合警告
            if train_auc - test_auc > 0.1:
                tqdm.write("⚠️  警告: 可能過擬合 (Train AUC - Test AUC = {:.3f})".format(train_auc - test_auc))
            
            if test_auc >= best_auc:
                save_best_weights(model_in, uuid_, task=task)
                best_auc = test_auc
                tqdm.write(f"🎯 New best AUC: {test_auc}")
        else:
            # 不評估時，重複使用上次的準確率
            eva_fig.append(eva_fig[-1] if eva_fig else 0)
        
        # 每 10 個 epoch 保存權重和圖表
        if i % 10 == 0 and i != 0:
            save_weights(model_in, uuid_, epoch=i, fold=k_folds, task=task)
            plt.clf()  # 清除舊圖
            plt.plot(range(1, i+1), loss_fig, label='Train Loss')
            plt.plot(range(1, i+1), eva_fig, label='Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.title('Training Progress')
            os.makedirs("../figures/", exist_ok=True)
            plt.savefig("../figures/" + uuid_ + '_eva.png')
            plt.close()  # 關閉圖表釋放記憶體


def train_camull(ld_helper, k_folds=1, model=None, epochs=40, model_name='nnmamba'):
    '''The function for training the camull network'''
    task = ld_helper.get_task_string()
    # uuid_ = "Densenet_1717_{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    uuid_ = "nnMamba_{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
    print(uuid_)

    model_cop = model

    for k_ind in range(k_folds):

        if model_cop is None:
            model = build_arch(model_name)
        else:
            model = model_cop

        train_dl = ld_helper.get_train_dl(k_ind)
        test_dl = ld_helper.get_test_dl(k_ind)
        train_loop(model, train_dl, test_dl, epochs, uuid_, k_folds=k_ind + 1, task=task)
        print("Completed fold {}/{}.".format(k_ind, k_folds))

    return uuid_


def main():
    setup_seed(42)

    # NC v AD
    ld_helper = LoaderHelper(task=Task.NC_v_AD)
    model_uuid = train_camull(ld_helper, epochs=100)
    evaluate_model(DEVICE, model_uuid, ld_helper)

    # transfer learning for pMCI v sMCI
    model_uuid = "xxx"
    ld_helper = LoaderHelper(task=Task.sMCI_v_pMCI)
    pth = f'../weights/NC_v_AD/{model_uuid}/best_weight.pth'
    model_name = 'nnmamba'
    model = build_arch(model_name)
    model.load_state_dict(torch.load(pth, map_location=DEVICE))
    # model = load_model(pth)
    uuid = train_camull(ld_helper, model=model, epochs=10)
    evaluate_model(DEVICE, uuid, ld_helper)

def eval():
    model_uuid = "xxx" 
    # ld_helper = LoaderHelper(task=Task.NC_v_AD)
    ld_helper = LoaderHelper(task=Task.sMCI_v_pMCI)
    evaluate_model(DEVICE, model_uuid, ld_helper)


if __name__ == "__main__":
    # main()
    eval()
