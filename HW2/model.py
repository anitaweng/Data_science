import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, dim_emb, ):
        super(Self_Attn,self).__init__()
        
        self.query = LinearNorm(dim_emb, dim_emb)
        self.key = LinearNorm(dim_emb, dim_emb)
        self.value = LinearNorm(dim_emb, dim_emb)

        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( dim_emb )
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        #print(x.shape)
        #x = x.unsqueeze(1)
        proj_query  = self.query(x) # B * 1 *dim_emb
        #print(proj_query.shape)
        proj_key =  self.key(x) # B * 1 *dim_emb
        #print(proj_key.shape)
        energy =  torch.bmm(proj_query.transpose(1,2),proj_key)
        #print(energy.shape)
        #energy =  torch.matmul(proj_query.permute(0,2,1), proj_key) # B * dim_emb * dim_emb
        attention = self.softmax(energy) # B * dim_emb * dim_emb
        #print(attention.shape)
        proj_value = self.value(x) # B * 1 *dim_emb
        #print(proj_value.shape)
        out =  torch.bmm(proj_value,attention.transpose(1,2))
        #print(out.shape)
        #out = torch.matmul(proj_value, attention.permute(0,2,1))
        #out = out.squeeze(1)
        
        #out = self.gamma*out + x
        return out, attention


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Classifier(nn.Module):
    """Encoder module:
    """
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=10, num_layers=2, rnn_type='LSTM',final_hidden_state = 1):
        super(Classifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.final_hidden_state = final_hidden_state
        #self.attn = Self_Attn(self.hidden_dim)

        #Define the initial linear hidden layer
        #self.init_linear = nn.Linear(self.input_dim, self.input_dim)

        # Define the LSTM layer 300 128 2 
        self.lstm = eval('nn.' + rnn_type)(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True) #B * 256
        '''self.lstm1 = eval('nn.' + rnn_type)(self.hidden_dim * 2, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = eval('nn.' + rnn_type)(self.hidden_dim * 2, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=True)'''
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim *2, output_dim) #* 2 #128 50 1 
        self.linear2 = nn.Linear(50, output_dim)
        self.s = nn.Sigmoid()

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        '''return (torch.randn(self.num_layers, self.batch_size, self.hidden_dim),
                torch.randn(self.num_layers, self.batch_size, self.hidden_dim))'''
        return torch.randn(self.num_layers*2, self.batch_size, self.hidden_dim)
        
    '''def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]'''

    def forward(self, input):
        # have shape (batch_size, num_layers, hidden_dim).
        '''hidden = self.init_hidden().to(torch.device('cuda'))
        c = self.init_hidden().to(torch.device('cuda'))
        lstm_out, (h, c) = self.lstm(input,(hidden,c))'''
        '''lstm_out, (h, c) = self.lstm1(lstm_out,(hidden,c))
        lstm_out, (h, c) = self.lstm2(lstm_out,(hidden,c))'''
        '''attn_output, attention = self.attention_net(lstm_out, h)
        attn_output = self.out(attn_output)'''
        '''print('lstm_out:'+str(lstm_out.shape))
        attn_output, attn_map = self.attn(lstm_out)
        #print('attn_output:'+str(attn_output.shape))
        attn_output = self.linear(attn_output)
        #print('linear:'+str(attn_output.shape))
        attn_output = self.linear2(attn_output.squeeze(2).unsqueeze(1)).squeeze()
        #print('linear2:'+str(attn_output.shape))
        
        
        return 5.0 * self.s(attn_output)#, attn_map'''
        
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        hidden = self.init_hidden().to(torch.device('cuda'))
        c = self.init_hidden().to(torch.device('cuda'))
        lstm_out, (h, c) = self.lstm(input,(hidden,c))
        #lstm_out, (h, c) = self.lstm1(lstm_out,(hidden,c))
        #lstm_out, (h, c) = self.lstm2(lstm_out,(hidden,c))
        y_pred = self.linear(lstm_out)
        #y_pred = self.linear2(y_pred.view)
        #print(y_pred.shape)
        if y_pred.shape[0] == 1:
            y_pred = self.s(y_pred).squeeze(2)
            output = self.linear2(y_pred.unsqueeze(1)).squeeze()
        else:
            y_pred = self.s(y_pred).squeeze()
            output = self.linear2(y_pred.unsqueeze(1)).squeeze()
        
        return 5 * output
