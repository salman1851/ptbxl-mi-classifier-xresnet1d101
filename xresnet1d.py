import torch
import torch.nn as nn
from params import *

class conv(nn.Module):
    
    def __init__(self, in_ch, out_ch, k_size = 25, stride = 1, 
                 drop_r = None, zero_batch_norm = False, 
                 bias = False, use_act_fun = True,
                 act_fun = nn.ReLU()):
        
        assert k_size % 2 == 1, 'kernel size shall be odd number'
        super(conv, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, k_size, stride,
                                padding=(k_size-1)//2, bias=bias,)
        self.batch_norm = nn.BatchNorm1d(out_ch)
        nn.init.constant_(self.batch_norm.weight,
                          0. if zero_batch_norm else 1.)
        self.act_fun = act_fun
        self.drop_r, self.drop = drop_r, nn.Dropout(drop_r) if drop_r else None
        self.use_act_fun = use_act_fun
        
    def forward(self, x):
        
        x = self.conv1d(x)
        x = self.batch_norm(x)
        if self.use_act_fun:
            x = self.act_fun(x)
        if self.drop_r:
            x = self.drop(x)
        
        return x

class XResNetBlock(nn.Module):
    
    def __init__(self, expansion, in_ch, between_ch, k = 9, 
                 stride = 1, b_verbose = None,
                 act_fun = nn.ReLU()):
        
        assert expansion in [1,4] , 'expansion shall be 1 or 4'
        super(XResNetBlock, self).__init__()
        
        in_ch = in_ch * expansion
        out_ch = between_ch * expansion
        
        if expansion == 1:
            
            layers = [conv(in_ch, between_ch, 
                           k, stride=stride),
                      conv(between_ch, out_ch, k, 
                           zero_batch_norm=True, 
                           use_act_fun = False)]
        
        else:
            
            layers = [conv(in_ch, between_ch, 1),
                      conv(between_ch, between_ch, 
                           k, stride = stride,),
                      conv(between_ch, out_ch, 1,
                           zero_batch_norm=True, 
                           use_act_fun = False)]
        
        self.xres_block = nn.ModuleList(layers)
        
        self.res_conv = conv(in_ch,out_ch,1,use_act_fun=False
                             ) if in_ch != out_ch else None
        self.res_pool = nn.AvgPool1d(2, ceil_mode=True
                                     ) if stride != 1 else None
        self.act_fun = act_fun
        self.b_verbose = b_verbose if b_verbose else None

    def forward(self, x):
        
        identity = x
        
        for l in self.xres_block:
            x = l(x)
            print('res_torch_size:', x.shape) if self.b_verbose else None
        
        identity = self.res_pool(identity) if self.res_pool else identity
        identity = self.res_conv(identity) if self.res_conv else identity
        print('identity_torch_size:', x.shape) if self.b_verbose else None
        
        x += identity
        x = self.act_fun(x)
        
        return x

class ConcatPool(nn.Module):
    def __init__(self):
        super(ConcatPool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        return torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)


class XResNet1d(nn.Module):

    def __init__(self, expansion, num_layers, stem_k, block_k,
                 in_ch=IN_CH, c_out=NUM_CLASSES, model_drop_r=None,
                 fc_drop=None, original_f_number=False):

        super(XResNet1d, self).__init__()
        self.block_k = block_k

        # --- STEM ---
        stem_filters = [in_ch, 32, 32, 64]
        stem = [conv(stem_filters[i], stem_filters[i + 1], k_size=stem_k,
                      stride=2 if i == 0 else 1, drop_r=model_drop_r)
                for i in range(3)]
        self.stem = nn.ModuleList(stem)
        self.stem_pool = nn.MaxPool1d(3, stride=2, padding=1)

        # --- STAGES ---
        if original_f_number:
            widths = [64, 128, 256, 512]
        else:
            widths = [64, 64, 64, 64]
        block_filters = [64 // expansion] + widths

        block = [self.make_layers(expansion, block_filters[i], block_filters[i + 1],
                                   n_blocks=l, stride=1 if i == 0 else 2)
                 for i, l in enumerate(num_layers)]
        self.block = nn.ModuleList(block)

        # --- HEAD ---
        final_ch = block_filters[-1] * expansion
        self.head_pool = ConcatPool()
        self.fc1 = nn.Linear(final_ch * 2, 128)
        self.fc_batch_norm = nn.BatchNorm1d(128)
        self.act_fun = nn.ReLU()
        self.fc_drop = nn.Dropout(fc_drop) if fc_drop else None
        self.fc_out = nn.Linear(128, c_out)

    def make_layers(self, expansion, n_inputs, n_filters, n_blocks, stride):
        sub_block = [XResNetBlock(expansion,
                                   n_inputs if i == 0 else n_filters,
                                   n_filters, self.block_k,
                                   stride if i == 0 else 1)
                     for i in range(n_blocks)]
        return nn.Sequential(*sub_block)

    def forward(self, x):

        for layer in self.stem:
            x = layer(x)
        x = self.stem_pool(x)

        for stage in self.block:
            x = stage(x)

        x = self.head_pool(x)          # (batch, final_ch*2, 1)
        x = x.flatten(1)               # (batch, final_ch*2)
        x = self.fc1(x)
        x = self.fc_batch_norm(x)
        x = self.act_fun(x)
        if self.fc_drop:
            x = self.fc_drop(x)
        x = self.fc_out(x)

        return x


def xresnet1d101(**kwargs):
    return XResNet1d(4, [3, 4, 23, 3], stem_k=kwargs.pop('stem_k', 7),
                      block_k=kwargs.pop('block_k', 5), **kwargs)

##### sanity check #####
# model = xresnet1d101(in_ch=IN_CH, c_out=NUM_CLASSES, stem_k=STEM_K, block_k=BLOCK_K)
# x = torch.randn(BATCH_SIZE, IN_CH, 5000)
# out = model(x)
# print(out.shape)   # expect (64, 3)