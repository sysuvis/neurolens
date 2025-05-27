from dataset_v3 import *
from models import *
import numpy as np   
import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchmetrics
from tqdm import tqdm,trange
import argparse
import json
import time


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=17045, help='seed')
parser.add_argument('--depth', type=int, default=3, help='number of gat layers')
parser.add_argument('--dataset', type=str, default="hybrid", help='dataset')
args = parser.parse_args()


def inference(model_depth,dataset):
    data_module = Brain_DataModule(dataset=dataset,seed=args.seed)
    if dataset=="hybrid":
        model=GAT.load_from_checkpoint(f"nn/gnn.ckpt",num_graph_layers=model_depth)
        
    trainer=pl.Trainer()
    results = trainer.test(model, data_module)
    return results



if __name__ == '__main__':
    depth=args.depth
    dataset=args.dataset
    
    inference(depth,dataset)
    
    #print("done")
    
    
    
    