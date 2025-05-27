from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool,global_max_pool,global_sort_pool,global_add_pool
from torch_geometric.nn import aggr
import torch_geometric.nn as pyg_nn
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Tuple, Union
from torch_geometric.utils import to_dense_adj
from starnet import ConvBN,Block
from timm.models.layers import trunc_normal_
import pytorch_lightning as pl
from torch.optim import Adam,AdamW
import torchmetrics
import torch.optim as optim
import lpips
from torch_scatter import scatter

def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

# GNN 
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
               num_graph_layers, dropout_pct):

        super(GCN, self).__init__()
        self.num_graph_layers = num_graph_layers
        self.dropout_pct = dropout_pct
        self.output_dim = output_dim

        # convert manually crafted categorical features to continuous
        #self.encoder = AtomEncoder(hidden_dim)
        self.encoder=GCNConv(input_dim,hidden_dim)
        self.convs = nn.ModuleList()
        for i in range(num_graph_layers):
            self.convs.append(pyg_nn.GCNConv(hidden_dim, hidden_dim))

        self.bns = nn.ModuleList()
        for i in range(num_graph_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.clf_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr
        
        x = self.encoder(x,edge_index)
        for i in range(self.num_graph_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_pct)
            x = self.convs[-1](x, edge_index)

        x = pyg_nn.global_mean_pool(x, data.batch)
        x = self.clf_head(x)
        return x

class GAT(pl.LightningModule):
    def __init__(self, input_dim=105, hidden_dim=64, output_dim=4, 
               num_graph_layers=3, dropout_pct=0.5):

        super(GAT, self).__init__()
        self.num_graph_layers = num_graph_layers
        self.dropout_pct = dropout_pct
        self.output_dim = output_dim

        # convert manually crafted categorical features to continuous
        #self.mixer = nn.Linear(input_dim, hidden_dim)
        self.encoder=GATConv(input_dim,hidden_dim)
        self.convs = nn.ModuleList()
        for i in range(num_graph_layers):
            self.convs.append(pyg_nn.GATConv(hidden_dim, hidden_dim))

        self.bns = nn.ModuleList()
        for i in range(num_graph_layers - 1):
            self.bns.append(pyg_nn.GraphNorm(hidden_dim))

        self.mlp1=nn.Linear(hidden_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.mlp2=nn.Linear(hidden_dim,hidden_dim)
        self.mlp3=nn.Linear(hidden_dim,hidden_dim)
        
        self.clf_head = nn.Linear(hidden_dim, output_dim)
        #metric
        self.train_losses = []
        self.val_losses = []
        self.test_outputs = []
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass',num_classes=output_dim,average='macro')
        self.precision = torchmetrics.Precision(task='multiclass',num_classes=output_dim,average='macro')
        self.recall = torchmetrics.Recall(task='multiclass',num_classes=output_dim,average='macro')
        self.f1 = torchmetrics.F1Score(task='multiclass',num_classes=output_dim,average='macro')
        
        self.poolings = nn.ModuleList()
        for _ in range(2):
            self.poolings.append(pyg_nn.SAGPooling(hidden_dim))
            
        self.dti_mixer=nn.Linear(5,hidden_dim)
        self.geometric_mixer=nn.Linear(100,hidden_dim)
        self.mixer=nn.Linear(hidden_dim*2,hidden_dim)
        self.mixer2=nn.Linear(hidden_dim,hidden_dim)
        self.bn=nn.BatchNorm1d(hidden_dim)

    def forward(self, data):
        x, edge_index, batch, edge_arr = data.x, data.edge_index, data.batch, data.edge_arr
        
        x_aggregated = torch.zeros_like(x)
        row, col = edge_index
        edge_weighted_features = edge_arr
        scatter(edge_weighted_features, col, dim=0, reduce='mean',out=x_aggregated)
        x = x_aggregated
        
        x1=self.dti_mixer(x[:,:5])
        x2=self.geometric_mixer(x[:,5:])
        # x=F.relu6(x1)*x2
        x=self.mixer(torch.concat((x1,x2),dim=1))
        x=self.bn(x)
        x=F.relu6(x)
       
        #x_org = x = self.encoder(x, edge_index)
        x_org = x
        for i in range(self.num_graph_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_pct, training=self.training)
            

        x = self.convs[-1](x, edge_index)
        x = x + x_org
        
        out1 = self.mlp1(x)
        out2 = self.relu(out1)
        out3 = self.mlp2(out2)
        x = x + out3
        
        x, edge_index, edge_attr, batch, perm, score=self.poolings[0](x,edge_index,batch=batch)

        x_global = pyg_nn.global_mean_pool(x,batch)
        
        x = self.mlp3(x_global)
        x = F.relu(x)
        x = x + x_global
        x = self.clf_head(x)
        return x
    
    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = F.cross_entropy(out, batch.y)
        self.log('train_loss', loss)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        labels=batch.y
        loss = F.cross_entropy(out, labels)
        self.log('val_loss', loss)
        self.val_losses.append(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = F.cross_entropy(out, batch.y)
        preds = torch.argmax(out, dim=1)
        
        # Log metrics for the current step
        self.accuracy(preds, batch.y)
        self.precision(preds, batch.y)
        self.recall(preds, batch.y)
        self.f1(preds, batch.y)
        
        #self.log('test_loss', loss)
        self.log('test_acc', self.accuracy)
        self.log('test_precision', self.precision)
        self.log('test_recall', self.recall)
        self.log('test_f1', self.f1)
        
        return {"loss": loss, "preds": preds, "targets": batch.y}
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}      
    

# bundle encoder

class SinActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)
    
class AutoEncoder(pl.LightningModule):
    def __init__(self, act="sin", base_dim=32, depths=[3, 3, 12, 5], encode_dim=10, output_dim=400):
        super(AutoEncoder, self).__init__()
        self.in_channel = 32
        if act == "relu":
            self.act_layer = nn.ReLU6()
        elif act == "sin":
            self.act_layer = SinActivation()
        self.stem = nn.Sequential(
            ConvBN(1, self.in_channel, kernel_size=3, stride=2, padding=1),
            self.act_layer
        )
        self.stages = nn.ModuleList()
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            self.stages.append(nn.Sequential(down_sampler, self.act_layer))

        # Encoder
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_channel, encode_dim)
            
        )

        # Decoder
        self.decoder = nn.Sequential(
            self.act_layer,
            nn.Linear(encode_dim, encode_dim),
            self.act_layer,
            nn.Linear(encode_dim, output_dim),
            self.act_layer,
            nn.Unflatten(1, (1, 20, 20))  
        )
        self.train_losses = []
        self.val_losses = []
        self.lpips_metric = lpips.LPIPS(net='alex')

    def forward(self, x):
        x=torch.unsqueeze(x,1)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.encoder(x)
        x=self.decoder(x)
        x=torch.squeeze(x,1)
        return x
    
    def reconstruct(self, x):
        x=torch.unsqueeze(x,1)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded=torch.squeeze(decoded,1)
        return decoded
    
    def encode(self, x):
        x=torch.unsqueeze(x,1)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        encoded = self.encoder(x)
        return encoded

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        self.train_losses.append(loss.item())
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', test_loss)
        rmse = torch.sqrt(test_loss)
        psnr = torchmetrics.functional.peak_signal_noise_ratio(y_hat, y)
        y_hat_channel,y_channel=torch.unsqueeze(y_hat,1),torch.unsqueeze(y,1)
        ssim = torchmetrics.functional.structural_similarity_index_measure(y_hat_channel, y_channel)
        
        self.log('test_rmse', rmse)
        self.log('test_psnr', psnr)
        self.log('test_ssim', ssim)
        
        return test_loss
    

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
class StarNet(pl.LightningModule):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0,encode_dim=10, **kwargs):
        super().__init__()
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(1, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # encoder
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.in_channel, encode_dim)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ReLU6(),
            nn.Linear(encode_dim, 256),
            nn.ReLU6(),
            nn.Linear(256, 16*5*5),
            nn.ReLU6(),
            nn.Unflatten(1, (16, 5, 5)),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU6(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)
            
        )
        
        
        self.apply(self._init_weights)
        self.lpips_metric = lpips.LPIPS(net='alex')
        self.train_losses = []
        self.val_losses = []

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x=torch.unsqueeze(x,1)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.encoder(x)
        x=self.decoder(x)
        x=torch.squeeze(x,1)
        return x
    
    def encode(self,x):
        x=torch.unsqueeze(x,1)
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.encoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        self.train_losses.append(loss.item())
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', test_loss)
        rmse = torch.sqrt(test_loss)
        psnr = torchmetrics.functional.peak_signal_noise_ratio(y_hat, y)
        y_hat_channel,y_channel=torch.unsqueeze(y_hat,1),torch.unsqueeze(y,1)
        ssim = torchmetrics.functional.structural_similarity_index_measure(y_hat_channel, y_channel)
        
        self.log('test_rmse', rmse)
        self.log('test_psnr', psnr)
        self.log('test_ssim', ssim)
        
        return test_loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}



        