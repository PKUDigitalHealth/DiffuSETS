import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd


class TimeEmbedding(nn.Module):
    def __init__(self, number_of_diffusions, n_channels=1, dim_embed=64, dim_latent=128):
        super(TimeEmbedding, self).__init__()

        self.number_of_diffusions = number_of_diffusions
        self.n_channels = n_channels
        self.fc1 = nn.Linear(dim_embed, dim_latent)
        self.fc2 = nn.Linear(dim_latent, n_channels)
        self.dim_embed = dim_embed
        self.embeddings = nn.Parameter(self.embed_table())
        self.embeddings.requires_grad = False

    def embed_table(self):
        t = torch.arange(self.number_of_diffusions) + 1 
        half_dim = self.dim_embed // 2
        emb = 10.0 / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # (num_diffusion, dim_embd)
        return emb

    def forward(self, t):
        # t: (B)
        # emb: (B, dim_embed)
        emb = self.embeddings[t, :]
        out = self.fc1(emb)
        out = F.mish(out)
        out = self.fc2(out)
        # out: (B, n_channels)
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, channels, hidden_dim=16, num_heads=4):
        super(SelfAttention, self).__init__() 

        self.hidden_dim = hidden_dim 
        self.to_q = nn.Linear(channels, hidden_dim) 
        self.to_k = nn.Linear(channels, hidden_dim) 
        self.to_v = nn.Linear(channels, hidden_dim) 
        
        self.to_out = nn.Linear(hidden_dim, channels)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels) 

    def forward(self, x):
        # x: (B, Seq_length, dim) i.e. (B, h * w, channels)
        x_norm = self.norm(x) 

        q = self.to_q(x_norm) 
        k = self.to_k(x_norm)
        v = self.to_v(x_norm)

        h, _ = self.attention(q, k, v) 
        h = self.to_out(h) 
        h = h + x 
        return h 


class CrossAttention(nn.Module):
    def __init__(self, vector_size, hidden_dim=16, num_heads=4, text_embed_dim=1536, value_embed_dim=1):
        # vector_size: encoding_dim, i.e. channels
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(vector_size, hidden_dim) 
        self.to_k = nn.Linear(vector_size, hidden_dim) 
        self.to_v = nn.Linear(vector_size, hidden_dim) 

        self.to_out = nn.Linear(hidden_dim, vector_size) 

        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.normalization = nn.LayerNorm(vector_size)
        self.value_combiner = nn.Linear(text_embed_dim + 3 * value_embed_dim, vector_size)
        self.condition_normalization = nn.LayerNorm(vector_size)

    def forward(self, x, text_embed, dict):
        # x: (B, seq_length, vector_size) or (B, l, c) 
        x_norm = self.normalization(x)
        q = self.to_q(x_norm)

        combined_embeds = []
        
        if text_embed.shape:
            combined_embeds = text_embed
        else:
            print("err_text")

        for key, value in dict.items():
            combined_embeds = torch.cat([combined_embeds, value], dim=-1)

        # combined_embeds: (B, seq_length, vector_size)
        combined_embeds_tensor = self.value_combiner(combined_embeds)
        combined_embeds_tensor = combined_embeds_tensor.repeat(1, x.shape[1] // combined_embeds_tensor.shape[1], 1)
        combined_embeds_norm = self.condition_normalization(combined_embeds_tensor)

        k = self.to_k(combined_embeds_norm)
        v = self.to_v(combined_embeds_norm)

        h, _ = self.attention(q, k, v)
        h = self.to_out(h)
        h = h + x
        # h: (B, seq_length, vector_size)
        return h


class Block(nn.Module):
    def __init__(self, n_inputs, n_outputs, number_of_diffusions,
                 kernel_size=5, n_heads=4, hidden_dim=16, text_embed_dim=1536):
        super(Block, self).__init__()
        n_shortcut = int((n_inputs + n_outputs) // 2)# + 1
        self.pre_shortcut_convs = nn.Conv1d(n_inputs, n_shortcut, kernel_size, padding="same")# padding="same"
        self.shortcut_convs = nn.Conv1d(n_shortcut, n_shortcut, 1, padding="same")#padding="same"
        self.post_shortcut_convs = nn.Conv1d(n_shortcut, n_outputs, kernel_size, padding="same")#, padding="same"
        self.down = nn.Conv1d(n_outputs, n_outputs, 3, 2, padding=1)#nn.MaxPool1d(2)
        self.layer_norm1 = nn.GroupNorm(1, n_shortcut)
        self.layer_norm2 = nn.GroupNorm(1, n_outputs)
        self.res_conv = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        # self.text_embedding_layer = nn.Linear(text_embed_dim, dim)
        self.self_attention = SelfAttention(channels=n_shortcut, hidden_dim=hidden_dim, num_heads=n_heads)
        self.cross_attention = CrossAttention(vector_size=n_shortcut, hidden_dim=hidden_dim, num_heads=n_heads, text_embed_dim=text_embed_dim)
        self.attention_norm = nn.LayerNorm(n_shortcut) 
        
        self.time_emb = TimeEmbedding(number_of_diffusions, n_inputs)

    def forward(self, x, t, text_embed, conditon):
        # x: (B, C_in, L)
        initial_x = x
        t = self.time_emb(t).unsqueeze(-1)
        # t: (B, C_in, 1)
        x = x + t

        # shortcut: (B, C_short, L)
        shortcut = self.pre_shortcut_convs(x)
        shortcut = self.layer_norm1(shortcut)
        shortcut = F.mish(shortcut)
        shortcut = self.shortcut_convs(shortcut)

        # shortcut: (B, L, C_short)
        shortcut = shortcut.transpose(-1, -2)
        shortcut = self.self_attention(shortcut)
        shortcut = self.cross_attention(shortcut, text_embed, conditon)
        shortcut = self.attention_norm(shortcut) 
        shortcut = shortcut.transpose(-1, -2)

        # out: (B, C_out, L)
        out = self.post_shortcut_convs(shortcut)
        out = self.layer_norm2(out)
        out = F.mish(out)
        out = (out + self.res_conv(initial_x))# / math.sqrt(2.0)
        return out


# modified to 2D
class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, number_of_diffusions, 
                 kernel_size=5, n_heads=None, hidden_dim=None, text_embed_dim=1536):
        super(DownsamplingBlock, self).__init__()
        self.down = nn.Conv1d(in_channels=n_outputs, out_channels=n_outputs, kernel_size=3, stride=2, padding=1)
        self.block = Block(n_inputs, n_outputs, number_of_diffusions, kernel_size=kernel_size, n_heads=n_heads, hidden_dim=hidden_dim, text_embed_dim=text_embed_dim)

    def forward(self, x, t, text_embed, conditon):
        h = self.block(x, t, text_embed, conditon)
        # DOWNSAMPLING
        out = self.down(h)
        return h, out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, number_of_diffusions,
                 kernel_size=5, up_dim=None, n_heads=None, hidden_dim=None, text_embed_dim=1536):
        super(UpsamplingBlock, self).__init__()
        self.block = Block(n_inputs, n_outputs, number_of_diffusions, kernel_size=kernel_size, n_heads=n_heads, hidden_dim=hidden_dim, text_embed_dim=text_embed_dim)

        if up_dim is None:
            self.up = nn.ConvTranspose1d(n_inputs // 2, n_inputs // 2, kernel_size=4, stride=2, padding=1)#padding=1
        else:
            self.up = nn.ConvTranspose1d(up_dim, up_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x, h, t, text_embed, conditon):
        x = self.up(x) 
        if h is not None:
            x = torch.cat([x, h], dim=1)
        out = self.block(x, t, text_embed, conditon)
        return out


class BottleneckNet(nn.Module):
    def __init__(self, n_channels, number_of_diffusions,
                 kernel_size=3, n_heads=4, hidden_dim=16, text_embed_dim=1536):
        super(BottleneckNet, self).__init__()
        self.time_emb = TimeEmbedding(number_of_diffusions, n_channels)        
        self.bottleneck_conv1 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv1_2 = nn.Conv1d(n_channels, n_channels , kernel_size=kernel_size, padding="same")
        self.bottleneck_conv2 = nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding="same")
        self.bottleneck_layer_norm1 = nn.GroupNorm(1, n_channels)
        self.bottleneck_layer_norm2 = nn.GroupNorm(1, n_channels)
        self.attention_norm = nn.LayerNorm(n_channels) 

        self.self_attention = SelfAttention(channels=n_channels, hidden_dim=hidden_dim, num_heads=n_heads)
        self.cross_attention = CrossAttention(n_channels, num_heads=n_heads, hidden_dim=hidden_dim, text_embed_dim=text_embed_dim)

    def forward(self, x, t, text_embed, conditon):
        out = x
        tt = self.time_emb(t).unsqueeze(-1)
        out = out + tt

        out = self.bottleneck_conv1(out)
        out = self.bottleneck_layer_norm1(out)
        out = F.mish(out)
        out = self.bottleneck_conv1_2(out)

        out = out.transpose(-1, -2)
        out = self.self_attention(out)
        out = self.cross_attention(out, text_embed, conditon)
        out = self.attention_norm(out) 
        out = out.transpose(-1, -2)

        #contional_emb = self.cond_emb(condition)
        #out = out + contional_emb

        out = self.bottleneck_conv2(out)
        out = self.bottleneck_layer_norm2(out)
        out = F.mish(out)
        
        out = (x + out) #/ math.sqrt(2)
        return out


class ECGconditional(nn.Module):
    def __init__(self, number_of_diffusions, kernel_size=3, num_levels=5, n_channels=4, text_embed_dim=1536):
        super(ECGconditional, self).__init__()

        self.num_levels = num_levels
        input_channels_list = []
        output_channels_list = []
        n_heads = 8

        for i in range(num_levels - 1):
            input_channels_list.append(n_channels * 2**i)  
        for i in range(num_levels - 1):    
            x = 2 * input_channels_list[num_levels - i - 2]
            input_channels_list.append(x)
                
        for i in range(num_levels - 1):
            output_channels_list.append(2 * input_channels_list[i]) 
        for i in range(num_levels - 1):    
            x = output_channels_list[num_levels - i - 2] // 2 
            output_channels_list.append(x)

        for i in range(num_levels - 2):
            k = 2 * (num_levels - 1) - i - 1
            input_channels_list[k] += output_channels_list[i]

        n_hidden_state_list = [channel * 2 for channel in input_channels_list]
        # print(input_channels_list)
        # print(n_hidden_state_list)

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)
        self.downsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = nn.ModuleList()

        for i in range(self.num_levels - 1):
            self.downsampling_blocks.append(
                DownsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                            number_of_diffusions=number_of_diffusions,
                            kernel_size=kernel_size, n_heads=n_heads, 
                            hidden_dim=n_hidden_state_list[i], text_embed_dim=text_embed_dim))

        self.bottelneck = BottleneckNet(n_channels=input_channels_list[num_levels],
                                        number_of_diffusions=number_of_diffusions, n_heads=4, hidden_dim=32, text_embed_dim=text_embed_dim)

        i = self.num_levels - 1
        self.upsampling_blocks.append(
            UpsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                                  number_of_diffusions=number_of_diffusions,
                                  kernel_size=kernel_size, up_dim=input_channels_list[i],
                                  n_heads=n_heads, hidden_dim=n_hidden_state_list[i], text_embed_dim=text_embed_dim))
        for i in range(self.num_levels, 2*self.num_levels - 2):
            self.upsampling_blocks.append(
                UpsamplingBlock(n_inputs=input_channels_list[i], n_outputs=output_channels_list[i],
                                  number_of_diffusions=number_of_diffusions,
                                  kernel_size=kernel_size, n_heads=n_heads, hidden_dim=n_hidden_state_list[i], text_embed_dim=text_embed_dim))


        self.output_conv = nn.Sequential(nn.Conv1d(output_channels_list[-1], n_channels, 3, padding="same"), nn.Mish(),
                                         nn.Conv1d(n_channels, n_channels, 1, padding="same"))

    def forward(self, x, t, text_embed, condition):
        '''
        '''
        shortcuts = []
        out = x

        # DOWNSAMPLING BLOCKS
        for block in self.downsampling_blocks:
            h, out = block(out, t, text_embed, condition)
            shortcuts.append(h)
            # print(out.shape)
        del shortcuts[-1]
        #out = self.downsampling_blocks[-1](out)

        # BOTTLENECK CONVOLUTION
        out = self.bottelneck(out, t, text_embed, condition) 
        # print(out.shape)      

        # UPSAMPLING BLOCKS
        out = self.upsampling_blocks[0](out, None, t, text_embed, condition)
        # print(out.shape)
        for idx, block in enumerate(self.upsampling_blocks[1:]):
            out = block(out, shortcuts[-1-idx], t, text_embed, condition)
            # print(out.shape)

        # OUTPUT CONV
        out = self.output_conv(out)
        return out


if __name__ == '__main__':

    unet = ECGconditional(1000)
    batch_size = 64 
    vae_latent = torch.randn((batch_size, 4, 128))
    t = torch.randperm(1000)[:batch_size]
    text_embed = torch.randn((batch_size, 1, 1536))
    condition = {'gender': torch.randn((batch_size, 1, 1)), 
                    'age': torch.randn((batch_size, 1, 1)), 
                    'heart rate': torch.randn((batch_size, 1, 1))}
    
    output = unet(vae_latent, t, text_embed, condition)
