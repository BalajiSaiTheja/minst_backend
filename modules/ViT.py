import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as dataloader
import torch.nn as nn
import numpy as np
import math
class PositionalEmbedding(nn.Module):
    def __init__(self,seq_len,embd_dim):
        super().__init__()
        embs = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0,embd_dim,2)*(-torch.log(torch.tensor(10000)) / embd_dim)
        )
        pos = torch.zeros(seq_len,embd_dim)
        pos[:,0::2] = torch.sin(embs*div_term)
        pos[:,1::2] = torch.cos(embs*div_term)

        self.register_buffer("pe",pos.unsqueeze(0))

    def forward(self,x):
        return x + self.pe[:,:x.size(1)]
    

img_dim = 28
class PatchEmbedding(nn.Module):
    def __init__(self,in_channels=1,patch_dim=7,embd_dim=512):
        super().__init__()
        self.patch_size = patch_dim
        self.patch_dim = patch_dim*patch_dim*in_channels
        self.linear = nn.Linear(self.patch_dim,embd_dim)
        self.cls_token = nn.Parameter(torch.rand(1,1,embd_dim))
        self.pos_embd = PositionalEmbedding((img_dim//patch_dim)**2+1,embd_dim)
        self.embd_dim = embd_dim
        
    def forward(self,x):
        """
            input dim = (64,1,28,28) 
            output_diim = (64,16,embd_dim)
        """
        #(64,1,4,4,7,7)
        batch_size,channels,height,width = x.shape
      
        imgs = x.unfold(2,self.patch_size,self.patch_size).unfold(3,self.patch_size,self.patch_size)
        
        imgs = imgs.contiguous().view(batch_size,channels,-1,self.patch_size,self.patch_size)
      
        imgs = imgs.view(batch_size,-1,self.patch_dim)
      
        embd = self.linear(imgs)

        cls_token = self.cls_token.expand(batch_size,-1,-1)

        x = torch.cat((cls_token,embd),dim=1)
        x = x + self.pos_embd(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self,embd_dim=512,eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embd_dim))
        self.beta = nn.Parameter(torch.zeros(embd_dim))
        self.eps = eps


    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)
        var  = x.var(dim=-1,keepdim=True,unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class MultiheadAttention(nn.Module):
    def __init__(self,heads=8,embd_dim=512):
        super().__init__()
        assert embd_dim%heads==0
        self.d_k = embd_dim // heads
        self.heads = heads
        self.w_q = nn.Linear(embd_dim,embd_dim)
        self.w_k = nn.Linear(embd_dim,embd_dim)
        self.w_v = nn.Linear(embd_dim,embd_dim)
        self.w_o = nn.Linear(embd_dim,embd_dim)

    def forward(self,x):
        b,seq_len,embd_len = x.shape
     #   print(x.shape)
        q_ = self.w_q(x)
    #    print(q_.shape)
        qh = q_.view(b,seq_len,self.heads,self.d_k).transpose(1,2)
      #  print(qh.shape)
        k_ = self.w_k(x)
        kh = k_.view(b,seq_len,self.heads,self.d_k).transpose(1,2)
        v_ = self.w_v(x)
        vh = v_.view(b,seq_len,self.heads,self.d_k).transpose(1,2)
        
        product = qh@(kh.transpose(-2,-1))
       # print(product.shape)
        atten = F.softmax(product / math.sqrt(self.d_k), dim=-1)
       # print(atten.shape)
        atten = atten @ vh
       # print(atten.shape)
        atten_merged = atten.transpose(1,2).contiguous().view(b,seq_len,embd_len)
      #  print(atten_merged.shape)
        return self.w_o(atten_merged)
        

class MLP(nn.Module):
    def __init__(self,hidden_factor=2,embd_dim = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embd_dim,hidden_factor*embd_dim),
            nn.ReLU(),
            nn.Linear(hidden_factor*embd_dim,embd_dim)
        )

    def forward(self,x):
        return self.mlp(x)

class Encoder(nn.Module):
    def __init__(self,in_channels=1,patch_dim=7,embd_dim=512,eps=1e-5,heads=8,hidden_factor=2,dropout=0.1):
        super().__init__()
        
        self.norm1 = LayerNorm(embd_dim,eps)
        self.norm2 = LayerNorm(embd_dim,eps)
        self.mha = MultiheadAttention(heads,embd_dim)
        self.mlp = MLP(hidden_factor,embd_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        
        norm1= self.norm1(x)
        attn = self.mha(norm1)
        x = x + self.dropout(attn)

        norm2 = self.norm2(x)
        mlp = self.mlp(norm2)
        x = x + self.dropout(mlp)

        return x

class ViT(nn.Module):
    def __init__(self,no_blocks=4,in_channels=1,patch_dim=7,embd_dim=512,
                 eps=1e-5,heads=8,hidden_factor=2,num_classes=10,dropout=0.1):
        super().__init__()
        self.p = PatchEmbedding(in_channels,patch_dim,embd_dim)
        self.encoder = nn.ModuleList([
          Encoder(
            in_channels,
            patch_dim,
            embd_dim,
            eps,
            heads,
            hidden_factor,
            dropout
        )
            for _ in range(no_blocks)
        ])
        self.head = nn.Sequential(
             nn.Linear(embd_dim, embd_dim),
             nn.ReLU(),
             nn.Linear(embd_dim, 10),    
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.p(x)
        for enc in self.encoder:
            x = enc(x)
        cls_token_output = x[:, 0]
        x = cls_token_output
        x = self.dropout(x)
        logits = self.head(x)       

        return logits
transformation_operation = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])