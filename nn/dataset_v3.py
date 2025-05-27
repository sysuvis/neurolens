import os.path as osp
import torch
from torch_geometric.data import Dataset,DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd 
import numpy as np 
from p_tqdm import *
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import random_split
    

class Hybrid_Dataset(InMemoryDataset):
    def __init__(self, root="./", transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self): 
        #return [f'data_{i}.pt' for i in range(198)]   
        return ["hybrid_features.dataset"]
        
    def len(self):
        return len(self.data)
    
    def get(self,index):
        return self.data[index]
    
    def download(self):
        pass
    
    def process(self):
        encoded_vecs=np.load('./data/AE_encode/encoded_vecs_0.npy')
        for i in range(1,10):
            tmp=np.load(f'./data/AE_encode/encoded_vecs_{i}.npy')
            encoded_vecs=np.hstack((encoded_vecs,tmp))
            
        with open("./data/AE_data_names.txt", "r") as file:
            item_names = [line.strip() for line in file.readlines()]
        
        # start process graph
        data_list=[]
        feature_diffusion=pd.read_excel('./data/feature_diffusion.xlsx')
        # for each subject(graph)
        for index4subject,row in tqdm(feature_diffusion.iterrows(),desc="processing graphs"):
            subject=row[0]
            cnt=1
            edges,edge_arr=[],[]
            nodes = [[0.0] * 105 for _ in range(70)]
            #for each bundle
            for i in range(70):
                for j in range(i+1,70):
                    bundle_item=subject+"+"+str(i+1)+"_"+str(j+1)
                    dti_values=[float(x) for x in row[cnt].split(',')]
                    try:
                        index = item_names.index(bundle_item)
                    except ValueError:
                        index= -1
                    if index!=-1 and any(value != 0 for value in dti_values):
                        edges.append([i,j])
                        geometric_values=encoded_vecs[index]
                        values=geometric_values.tolist()+dti_values
                        edge_arr.append(values)
                    cnt+=1
            edges=torch.tensor(edges)
            edges=torch.transpose(edges,dim0=0,dim1=1)
            edge_arr=torch.tensor(edge_arr)
            label=torch.tensor([row[-1]])
            nodes=torch.tensor(nodes)
            
            data=Data(x=nodes,edge_index=edges,edge_arr=edge_arr,y=label)
            data_list.append(data)
        
        #data,slices=self.collate(data_list)
        torch.save(data_list,self.processed_paths[0])

class Brain_DataModule(pl.LightningDataModule):
    def __init__(self,seed, val_ratio=0.2, batch_size=200,dataset="hybrid"):
        super().__init__()
        self.seed = seed
        if dataset=="hybrid":
            self.dataset = Hybrid_Dataset()
        self.test_ratio = val_ratio
        self.batch_size = batch_size
        self.setup()

    def setup(self, stage=None):
        dataset_len = len(self.dataset)
        test_size = int(self.test_ratio * dataset_len)
        train_size = dataset_len - test_size

        generator = torch.Generator().manual_seed(self.seed)

        self.train_dataset, self.test_dataset = random_split(
            self.dataset,
            [train_size, test_size],
            generator=generator
        )
        
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

if __name__=="__main__":
    data=Hybrid_Dataset()
    
   