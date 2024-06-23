# Torch
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 4, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads

        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    # def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    def __init__(self, num_patch, patch_size, P=8 , dim=32, depth=3, heads=4, mlp_dim = 128, dropout = 0.2, emb_dropout = 0., pool = 'cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.P = P
        output_class = number_gesture + int(class_rest==True)

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(int(patch_size/self.P), 4*dim),
            nn.ReLU(),
            nn.Linear(4*dim, dim),
        )
        

        # self.to_patch_embedding = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, int(num_patch*P) + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        dim_head = int(dim/heads)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, output_class)

    def forward(self, x):

        x = x.permute(0,2,1)  # shape: (B, W, C) -> shape: (B, C, W) 
        batch_size, num_patch, patch_size = x.shape
        
        x = x.reshape(batch_size, -1, int(patch_size/self.P))
        x = self.to_patch_embedding(x)
        

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViT_TNet(nn.Module):
    def __init__(self, window_size, num_channel, F=5, P=1 , dim=144, depth=1, heads=8, mlp_dim = 720, dropout = 0.4, emb_dropout = 0.4, pool = 'cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.conv_max_pool_factor = 4

        self.F = F  # F = output channels / input channels, in conv1d => like "F"iltering along time axis
        self.P = P  # P = num_patch / number_channel = patch_size * window_size, a scale factor to reshape the patch embedding
        self.num_channel = num_channel
        self.window_size = window_size
        self.patch_size = window_size // (P*self.conv_max_pool_factor)
        self.num_patch = num_channel * P * self.F
        
        output_class = number_gesture + int(class_rest==True)

        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_channel),
            nn.Conv1d(num_channel, num_channel*F, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(self.conv_max_pool_factor),
        )
        
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.patch_size, dim)
            # nn.Linear(self.patch_size, 4*dim),
            # nn.ReLU(),
            # nn.Linear(4*dim, dim),
        )

        # self.to_patch_embedding = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        dim_head = int(dim/heads)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, output_class)

    def get_class_emb(self, x):
        # input shape: (batch_size, window_size, num_channel) = (B, W, C)
        batch_size = x.shape[0]

        x = x.permute(0,2,1)  # shape: (B, W, C) -> shape: (B, C, W) 
        x = self.conv(x) # shape: (batch_size, F*num_channel, window_size/self.conv_max_pool_factor) = (B, F*C, W/self.conv_max_pool_factor)
        
        x = x.reshape(batch_size, -1, self.patch_size)
        x = self.to_patch_embedding(x)
        

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return x
    
    def forward(self, x):

        x = self.get_class_emb(x)
        x = self.to_latent(x)
        
        return self.mlp_head(x)

class ViT_FNet(nn.Module):
    def __init__(self, window_size, num_channel, F=5, P=10, Q=10, dim=144, depth=1, heads=8, mlp_dim = 720, dropout = 0.4, emb_dropout = 0.4, pool = 'cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.conv_max_pool_factor = 4

        self.F = F  # F = output channels / input channels, in conv1d => like "F"iltering along time axis
        self.P = P  # P = num_patch / number_channel = patch_size * window_size, a scale factor to reshape the patch embedding
        self.Q = Q
        self.num_channel = num_channel
        self.window_size = window_size
        self.patch_size = self.P * self.Q
        self.num_patch = (self.num_channel*self.F//self.P) * (self.window_size//(self.conv_max_pool_factor*self.Q))
        
        output_class = number_gesture + int(class_rest==True)

        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_channel),
            nn.Conv1d(num_channel, num_channel*F, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(self.conv_max_pool_factor),
        )
        
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.patch_size, dim)
            # nn.Linear(self.patch_size, 4*dim),
            # nn.ReLU(),
            # nn.Linear(4*dim, dim),
        )

        # self.to_patch_embedding = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        dim_head = int(dim/heads)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, output_class)

    def get_class_emb(self, x):
        # input shape: (batch_size, window_size, num_channel) = (B, W, C)
        batch_size = x.shape[0]

        x = x.permute(0,2,1)  # shape: (B, W, C) -> shape: (B, C, W) 
        x = self.conv(x) # shape: (batch_size, F*num_channel, window_size/self.conv_max_pool_factor) = (B, F*C, W/self.conv_max_pool_factor)
        
        # x = x.reshape(batch_size, -1, self.patch_size)
        x = rearrange(x, 'b (c p) (w q) -> b (c w) (p q)', p = self.P, q = self.Q)
        x = self.to_patch_embedding(x)
        

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return x
    
    def forward(self, x):

        x = self.get_class_emb(x)
        x = self.to_latent(x)
        
        return self.mlp_head(x)

        
class ViT_TraHGR(nn.Module):
    def __init__(self, window_size, num_channel, F=5, Pt=1, Pf=5, Qf=10, dim=144, depth=1, heads=8, mlp_dim = 720, dropout = 0.2, emb_dropout = 0.2, pool = 'cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        output_class = number_gesture + int(class_rest==True)

        self.ViT_TNet = ViT_TNet(window_size, num_channel, F=F, P=Pt , dim=dim, depth=depth, heads=heads, mlp_dim = mlp_dim, dropout = dropout, emb_dropout = emb_dropout, pool = pool, number_gesture=number_gesture, class_rest=class_rest)
        self.ViT_FNet = ViT_FNet(window_size, num_channel, F=F, P=Pf, Q=Qf, dim=dim, depth=depth, heads=heads, mlp_dim = mlp_dim, dropout = dropout, emb_dropout = emb_dropout, pool = pool, number_gesture=number_gesture, class_rest=class_rest)

        self.mlp_head_TNet = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_class)
        )

        self.mlp_head_FNet = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_class)
        )

        self.mlp_head_TraHGR = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_class)
        )


    def forward(self, x):
            
        z_TNet = self.ViT_TNet.get_class_emb(x) # (num_batch, dim, 1)
        z_FNet = self.ViT_FNet.get_class_emb(x) # (num_batch, dim, 1)

        return self.mlp_head_TNet(z_TNet), self.mlp_head_FNet(z_FNet), self.mlp_head_TraHGR(z_TNet+z_FNet)