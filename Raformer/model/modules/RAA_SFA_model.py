import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import numpy as np

class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding,
                 t2t_param):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

        self.f_h = int(
            (t2t_param['output_size'][0] + 2 * t2t_param['padding'][0] -
             (t2t_param['kernel_size'][0] - 1) - 1) / t2t_param['stride'][0] +
            1)
        self.f_w = int(
            (t2t_param['output_size'][1] + 2 * t2t_param['padding'][1] -
             (t2t_param['kernel_size'][1] - 1) - 1) / t2t_param['stride'][1] +
            1)

    def forward(self, x, b):
        feat = self.t2t(x)#torch.Size([1, 12, 128, 60, 108])
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, self.f_h, self.f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride,
                 padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.t2t = torch.nn.Fold(output_size=output_size,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)
        h, w = output_size
        self.bias = nn.Parameter(torch.zeros((channel, h, w),
                                             dtype=torch.float32),
                                 requires_grad=True)

    def forward(self, x, t):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = self.t2t(feat) + self.bias[None]
        return feat
    
class RAA_SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(RAA_SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        attention_weights = F.softmax(q @ k.transpose(-1, -2), dim=-1)
        return attention_weights

class SelfAttention2(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention2, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)

    def forward(self, x,y):
        q = self.query(x)
        k = self.key(y)
        attention_weights = F.softmax(q @ k.transpose(-1, -2), dim=-1)
        return attention_weights
    
class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                              padding=padding)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)
    
#######################
#初代windowattention
######################
class WindowAttention(nn.Module):
    def __init__(self, window_size, input_dim):
        super(WindowAttention, self).__init__()
        self.window_size = window_size
        self.attention = RAA_SelfAttention(input_dim)
        hidden = 512
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        channel = 256
        output_size = (60//2, 108//2)
        output_size2 = (60, 108)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'output_size': output_size2
        }
        self.sc = SoftComp(channel // 2, hidden, output_size, kernel_size,
                                stride, padding)
        
        self.ss = SoftSplit(channel // 2,
                            hidden,
                            kernel_size,
                            stride,
                            padding,
                            t2t_param=t2t_params)
        
        # decoder
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            )

    def window_partition(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B, T, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, T * self.window_size[0] * self.window_size[1], C)
        return windows
    
    
    
 
    def window_reverse(self,windows, T, H, W):
        """
        Args:
            windows: shape is (num_windows*B, T, window_size, window_size, C)
            window_size (tuple[int]): Window size
            T (int): Temporal length of video
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, T, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / self.window_size[0] / self.window_size[1]))
        x = windows.view(B, H // self.window_size[0], W // self.window_size[1], T,
                        self.window_size[0], self.window_size[1], -1)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
        return x

    def forward(self, x):
        res_x = x
        B, T, H, W, C = x.shape
        
        # Step 1: Window partition and compute mean
        windows = self.window_partition(x)#16, 540, 512
        
        window_means = windows.mean(dim=1)#16,512
        # Step 2: Compute attention weights
        attention_weights = self.attention(window_means)#16,16
        
        # Step 3: Compute attention scores (sum across the last dimension)
        attention_scores = attention_weights.sum(dim=-1)

        # Step 4: Select top-k windows based on the attention scores
        _, top_indices = torch.topk(attention_scores, k=attention_scores.size(0) // 4)

        selected_windows = windows[top_indices]
        
        y = self.window_reverse(selected_windows,T,H//2,W//2)
        
        trans_feat = self.sc(y, T)
        trans_feat = trans_feat.view(B, T, -1, 60//2, 108//2)#torch.Size([1, 12, 128, 60, 108])
        output = self.decoder(trans_feat.view(B * T, 128, 60//2, 108//2))  
        _, c, h, w = output.size()  
        trans_feat = self.ss(output.view(-1, c, h, w), B) + res_x
       

    
        return trans_feat



class RAA_SFA(nn.Module):
    def __init__(self, window_size, input_dim):
        super(RAA_SFA, self).__init__()
        self.window_size = window_size
        self.attention = RAA_SelfAttention(input_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5)) 
        self.beta = nn.Parameter(torch.tensor(0.5))  
        hidden = 512
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        channel = 256
        output_size = (60//2, 108//2)
        output_size2 = (60, 108)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'output_size': output_size2
        }
        self.sc = SoftComp(channel // 2, hidden, output_size, kernel_size,
                                stride, padding)
        
        self.ss = SoftSplit(channel // 2,
                            hidden,
                            kernel_size,
                            stride,
                            padding,
                            t2t_param=t2t_params)
        
        # decoder
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            )

    def window_partition(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B, T, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, T * self.window_size[0] * self.window_size[1], C)
        return windows
    
    def window_partition2(self, x):
        B, T, H, W, C = x.shape
        x = x.view(B, T, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, T ,self.window_size[0] * self.window_size[1], C)
        return windows
    
    def window_reverse(self,windows, T, H, W):
        """
        Args:
            windows: shape is (num_windows*B, T, window_size, window_size, C)
            window_size (tuple[int]): Window size
            T (int): Temporal length of video
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, T, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / self.window_size[0] / self.window_size[1]))
        x = windows.view(B, H // self.window_size[0], W // self.window_size[1], T,
                        self.window_size[0], self.window_size[1], -1)
        x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
        return x
    

    def forward(self, x):
        #RAA modluels
        #print(self.alpha,self.beta)
        B, T, H, W, C = x.shape
        res_x = x        
        # Step 1: Window partition and compute mean
        windows2 = self.window_partition2(x)
        _, num_T, num_w, _ = windows2.shape
        window_means2 = windows2.mean(dim=2)
        
        # Step 2: Compute attention weights
        attention_weights2 = self.attention(window_means2.view(-1, C))
        
        # Step 3: Compute attention scores (sum across the last dimension)
        attention_scores2 = attention_weights2.sum(dim=-1)
        
        # Step 4: Select top-k windows based on the attention scores
        _, top_indices2 = torch.topk(attention_scores2, k=attention_scores2.size(0) // 2)
        windows2 = windows2.view(-1, self.window_size[0]*self.window_size[1], C)
        
        selected_windows = windows2[top_indices2]

        #SFA modluels
        selected_windows = selected_windows.view(-1, num_T*num_w, C)
        
        
        y2 = self.window_reverse(selected_windows, T, H//2, W//2)
        

        trans_feat = self.sc(y2, T)
        
        trans_feat = trans_feat.view(B*2, T, -1, 60//2, 108//2)
        trans_feat = trans_feat.view(B, 2, T, -1, 60//2, 108//2)
        trans_feat = trans_feat.mean(dim=1)

        output = self.decoder(trans_feat.view(B * T, 128, 60//2, 108//2))  
        _, c, h, w = output.size()  
        trans_feat = self.alpha * self.ss(output.view(-1, c, h, w), B) + self.beta * res_x
        #trans_feat = self.ss(output.view(-1, c, h, w), B) + res_x
        return trans_feat
       
    
