import os
import csv
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data import Dataset_Pro
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def init_train(config):

    if config["model"] == "BWNET_LAGConv":
        from model.bwnet_lagconv_sota import BWNet, summaries
    elif config["model"] == "BWNET_DICNN":
        from model.bwnet_dicnn import BWNet, summaries
    elif config["model"] == "BWNET_FusionNet":
        from model.bwnet_fusionnet import BWNet, summaries
    elif config["model"] == "BWNET_LGPNet":
        from model.bwnet_lgpnet import BWNet, summaries

    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])
    random.seed(config["SEED"])
    cudnn.benchmark = False
    cudnn.deterministic = True

    device = torch.device(config["device"])

    if "wv3" in config["data_path_train"].lower():
        config["dataset"] = "wv3"
        config["num_bands"] = 8
    elif "gf2" in config["data_path_train"].lower():
        config["dataset"] = "gf2"
        config["num_bands"] = 4
    elif "qb" in config["data_path_train"].lower():
        config["dataset"] = "qb"
        config["num_bands"] = 4

    config["tensorboard_log_dir"] = os.path.join(config["tensorboard_log_dir"], config["dataset"], config["model"])
    config["loss_dir"] = os.path.join(config["loss_dir"], config["dataset"], config["model"])
    config["model_weights_dir"] = os.path.join(config["model_weights_dir"], config["dataset"], config["model"])
    config["train_loss_csv"] = os.path.join(config["loss_dir"], "train_loss.csv")
    config["val_loss_csv"] = os.path.join(config["loss_dir"], "val_loss.csv")
    os.makedirs(config["tensorboard_log_dir"], exist_ok=True)
    os.makedirs(config["loss_dir"], exist_ok=True)
    os.makedirs(config["model_weights_dir"], exist_ok=True)

    model = BWNet(ms_dim=config["num_bands"]).to(device)
    if config["model_resume_path"] is not None:
        model.load_state_dict(torch.load(config["model_resume_path"], map_location=device))

    if config["dataset"] == "wv3":
        summaries(model, input_size=[(1, 8, 16, 16), (1, 1, 64, 64)], grad=True)
    elif config["dataset"] == "gf2":
        summaries(model, input_size=[(1, 4, 16, 16), (1, 1, 64, 64)], grad=True)
    elif config["dataset"] == "qb":
        summaries(model, input_size=[(1, 4, 16, 16), (1, 1, 64, 64)], grad=True)

    train_set = Dataset_Pro(config["data_path_train"], config["dataset"])
    training_data_loader = DataLoader(dataset=train_set, num_workers=config["num_workers_train"], batch_size=config["batch_size"],
                                      shuffle=config["shuffle"], pin_memory=config["pin_memory"], drop_last=True)
    
    validate_set = Dataset_Pro(config["data_path_val"], config["dataset"])
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=config["num_workers_val"], batch_size=config["batch_size"],
                                      shuffle=config["shuffle"], pin_memory=config["pin_memory"], drop_last=True)

    return model, device, training_data_loader, validate_data_loader

def save_checkpoint(model, epoch, config):
    model_out_path = os.path.join(config["model_weights_dir"], f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)

def train(config):

    model, device, training_data_loader, validate_data_loader = init_train(config)
    criterion = nn.L1Loss().to(device)
    if config["save_band_loss"]:
        criterion_band = nn.L1Loss().to(device)
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': config["lr"]}], lr=config["lr"], weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config["step_size"], gamma=0.5, last_epoch=config["start_epoch"])

    writer = SummaryWriter(config["tensorboard_log_dir"])

    print('Start training...')

    if config["save_band_loss"]:
        with open(config["train_loss_csv"], mode='w', newline='') as train_file:
            writer_train_csv = csv.writer(train_file)
            writer_train_csv.writerow(['Epoch', 'Train Loss'] + [f'Train Band {i}' for i in range(config["num_bands"])])
        with open(config["val_loss_csv"], mode='w', newline='') as val_file:
            writer_val_csv = csv.writer(val_file)
            writer_val_csv.writerow(['Epoch', 'Val Loss'] + [f'Val Band {i}' for i in range(config["num_bands"])])

    for epoch in range(config["start_epoch"], config["epochs"], 1):
        model.train()
        epoch_train_loss = []
        if config["save_band_loss"]:
            band_train_loss = [0.0 for _ in range(config["num_bands"])]
        
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, pan, ms = batch[0].to(device), batch[3].to(device), batch[4].to(device)
            optimizer.zero_grad()
            output = model(ms, pan)
            loss = criterion(output, gt)
            epoch_train_loss.append(loss.item())

            if config["save_band_loss"]:
                for i in range(config["num_bands"]):
                    band_loss = criterion_band(output[:, i, :, :], gt[:, i, :, :]).item()
                    band_train_loss[i] += band_loss
            
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        t_loss = np.nanmean(np.array(epoch_train_loss))
        print(f'Epoch: {epoch}/{config["epochs"]}  training loss: {t_loss:.7f}')
        
        if config["save_band_loss"]:
            avg_train_band_loss = [band_train_loss[i] / len(training_data_loader) for i in range(config["num_bands"])]
            with open(config["train_loss_csv"], mode='a', newline='') as train_file:
                writer_train_csv = csv.writer(train_file)
                writer_train_csv.writerow([epoch, t_loss] + avg_train_band_loss)
            for i in range(config["num_bands"]):
                writer.add_scalar(f'Loss_train_band/band_{i}', avg_train_band_loss[i], epoch)

        writer.add_scalar('Loss/train', t_loss, epoch)

        with torch.no_grad():
            if epoch % config["log_interval"] == 0:
                model.eval()
                epoch_val_loss = []
                if config["save_band_loss"]:
                    band_val_loss = [0.0 for _ in range(config["num_bands"])]
                
                for iteration, batch in enumerate(validate_data_loader, 1):
                    gt, pan, ms = batch[0].to(device), batch[3].to(device), batch[4].to(device)
                    sr = model(ms, pan)
                    val_loss_item = criterion(sr, gt)
                    epoch_val_loss.append(val_loss_item.item())
                    
                    if config["save_band_loss"]:
                        for i in range(config["num_bands"]):
                            band_loss = criterion_band(sr[:, i, :, :], gt[:, i, :, :]).item()
                            band_val_loss[i] += band_loss
            
                v_loss = np.nanmean(np.array(epoch_val_loss))
                print(f'---------------validate loss: {v_loss:.7f}---------------')

                if config["save_band_loss"]:
                    avg_val_band_loss = [band_val_loss[i] / len(validate_data_loader) for i in range(config["num_bands"])]
                    with open(config["val_loss_csv"], mode='a', newline='') as val_file:
                        writer_val_csv = csv.writer(val_file)
                        writer_val_csv.writerow([epoch, v_loss] + avg_val_band_loss)
                    for i in range(config["num_bands"]):
                        writer.add_scalar(f'Loss_val_band/band_{i}', avg_val_band_loss[i], epoch)

                writer.add_scalar('Loss/val', v_loss, epoch)
        
        if epoch % config["ckpt_interval"] == 0:
            save_checkpoint(model, epoch, config)
    
    writer.close()

if __name__ == "__main__":
    config = {
        "SEED": 1,
        "lr": 2e-3,
        "ckpt_interval": 10,
        "epochs": 500,
        "start_epoch": 0,
        "batch_size": 64,
        "step_size": 150,
        "num_workers_train": 16,
        "num_workers_val": 16,
        "pin_memory": True,
        "shuffle": True,
        "log_interval": 10,
        "save_band_loss": False,
        "data_path_train": '',
        "data_path_val": '',
        "tensorboard_log_dir": 'tf-logs',
        "loss_dir": "loss_data",
        "model_weights_dir": "weights",
        "model_resume_path": None,
        "model": "BWNET_LGPNet",
        "device": "cuda",
    }
    
    train(config)