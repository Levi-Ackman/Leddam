import torch
import torch.nn as nn
from layers.RevIN import RevIN
from layers.Leddam import Leddam

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.revin=configs.revin
        self.revin_layer=RevIN(channel=configs.enc_in,output_dim=configs.pred_len)
        self.leddam=Leddam(configs.enc_in,configs.seq_len,configs.d_model,
                       configs.dropout,configs.pe_type,kernel_size=25,n_layers=configs.n_layers)
        
        self.Linear_main = nn.Linear(configs.d_model, configs.pred_len) 
        self.Linear_res = nn.Linear(configs.d_model, configs.pred_len)
        self.Linear_main.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / configs.d_model) * torch.ones([configs.pred_len, configs.d_model])) 
    def forward(self, inp):

        if self.revin:
            inp=self.revin_layer(inp)
        res,main=self.leddam(inp)
        main_out=self.Linear_main(main.permute(0,2,1)).permute(0,2,1)
        res_out=self.Linear_res(res.permute(0,2,1)).permute(0,2,1)
        pred=main_out+res_out
        if self.revin:
            pred=self.revin_layer.inverse_normalize(pred)
        return pred
