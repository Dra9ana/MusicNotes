import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image

def default_model_params(img_height, vocabulary_size):
    params = dict()
    params['img_height'] = img_height
    params['img_width'] = None
    params['batch_size'] = 16
    params['img_channels'] = 1
    params['conv_blocks'] = 4
    params['conv_filter_n'] = [32, 64, 128, 256]
    params['conv_filter_size'] = [ [3,3], [3,3], [3,3], [3,3] ]
    params['conv_pooling_size'] = [ [2,2], [2,2], [2,2], [2,2] ]
    params['rnn_units'] = 512
    params['rnn_layers'] = 2
    params['vocabulary_size'] = vocabulary_size
    return params

class CTC_CNN(nn.Module):
    def __init__(self,height,vocabulary_size):
        super().__init__()
        self.input_size = 2048 #(N, L, Hin)
        self.hidden_size = 256
        self.height = height
        self.width = None
        self.width_reduction = 1
        self.height_reduction = 1
        self.batch_size = 16
        #self.num_of_channels = 1
        self.num_of_channels = [1, 32, 64, 128, 256]
        self.conv_kernel_sizes= [ 3,3,3,3 ]
        self.pool_kernel_sizes =self.pool_kernel_strides= [2, 2, 2, 2]
        self.rnn_units = 512
        self.rnn_layers = 2
        self.vocabulary_size = vocabulary_size
        
        self.relu = nn.LeakyReLU()
        
        self.conv1 = nn.Conv2d(self.num_of_channels[0], self.num_of_channels[1], kernel_size=self.conv_kernel_sizes[0],padding='same')
        self.conv2 = nn.Conv2d(self.num_of_channels[1], self.num_of_channels[2], kernel_size=self.conv_kernel_sizes[1],padding='same')
        self.conv3 = nn.Conv2d(self.num_of_channels[2], self.num_of_channels[3], kernel_size=self.conv_kernel_sizes[2],padding='same')
        self.conv4 = nn.Conv2d(self.num_of_channels[3], self.num_of_channels[4], kernel_size=self.conv_kernel_sizes[3],padding='same')
        #(N, C, H, W)   ->  (N, C, H, W)

        self.bn1 = nn.BatchNorm2d(self.num_of_channels[1])
        self.bn2 = nn.BatchNorm2d(self.num_of_channels[2])
        self.bn3 = nn.BatchNorm2d(self.num_of_channels[3])
        self.bn4 = nn.BatchNorm2d(self.num_of_channels[4])
        #(N, C, H, W)   ->  (N, C, H, W)
        
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_kernel_sizes[0],stride=self.pool_kernel_strides[0])
        self.pool2 = nn.MaxPool2d(kernel_size=self.pool_kernel_sizes[1],stride=self.pool_kernel_strides[1])
        self.pool3 = nn.MaxPool2d(kernel_size=self.pool_kernel_sizes[2],stride=self.pool_kernel_strides[2])
        self.pool4 = nn.MaxPool2d(kernel_size=self.pool_kernel_sizes[3],stride=self.pool_kernel_strides[3])
        #(N, C, H, W)
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,num_layers=self.rnn_layers, dropout=0.5,batch_first=True,bidirectional=True)
        #(N, L, Hin)    ->  (N, L, 2*Hout)
        #(N, W, C*H)
        
        self.fully_connected=nn.Linear(in_features=2*self.hidden_size,out_features=self.vocabulary_size+1)
        #(N, L, vocab_size+1)

        self.softmax = nn.LogSoftmax(dim=2)
    
    
    def forward(self,x):
        #print("Ulaz1")
        #print(x[0,0,50:100,50:100])
        x = self.conv1(x)
        
        #print("conv1")
        #print(x)
        #Image.fromarray(x.to('cpu').detach().numpy()[0]*255).show()
        x = self.bn1(x)
        #print("batch norm1")
        #print(x)
        #Image.fromarray(x*255).show()
        x = self.relu(x)
        #print("relu1")
        #print(x)
        #Image.fromarray(x*255).show()
        x = self.pool1(x)
        #Image.fromarray(x*255).show()
        #print("pool1")
        #print(x)
        
    
        self.width_reduction = self.width_reduction * self.pool_kernel_sizes[0]
        self.height_reduction = self.height_reduction * self.pool_kernel_sizes[0]

        ###print(x.shape)
        #print("Ulaz2")
        #print(x[0,0,50:100,50:100])
        x = self.conv2(x)
        #Image.fromarray(x*255).show()
        #print("conv2")
        #print(x)
        x = self.bn2(x)
        #Image.fromarray(x*255).show()
        #print("batch norm2")
        #print(x)
        x = self.relu(x)
        #Image.fromarray(x*255).show()
        #print("relu2")
        #print(x)
        x = self.pool2(x)
        #Image.fromarray(x*255).show()
        #print("pool2")
        #print(x)
    
        self.width_reduction = self.width_reduction * self.pool_kernel_sizes[1]
        self.height_reduction = self.height_reduction * self.pool_kernel_sizes[1]

        ###print(x.shape)
        #print("Ulaz3")
        #print(x[0,0,50:100,50:100])
        x = self.conv3(x)
        #Image.fromarray(x*255).show()
        #print("conv3")
        #print(x)
        x = self.bn3(x)
        #Image.fromarray(x*255).show()
        #print("batch norm3")
        #print(x)
        x = self.relu(x)
        #Image.fromarray(x*255).show()
        #print("relu3")
        #print(x)
        x = self.pool1(x)
        #Image.fromarray(x*255).show()
        #print("pool3")
        #print(x)
    
        self.width_reduction = self.width_reduction * self.pool_kernel_sizes[2]
        self.height_reduction = self.height_reduction * self.pool_kernel_sizes[2]

        ###print(x.shape)
        #print("Ulaz4")
        #print(x[0,0,50:100,50:100])
        x = self.conv4(x)
        #Image.fromarray(x*255).show()
        #print("conv4")
        #print(x)
        x = self.bn4(x)
        #Image.fromarray(x*255).show()
        #print("batch norm4")
        #print(x)
        x = self.relu(x)
        #Image.fromarray(x*255).show()
        #print("relu4")
        #print(x)
        x = self.pool4(x)
        #Image.fromarray(x*255).show()
        #print("pool4")
        #print(x)
    
        self.width_reduction = self.width_reduction * self.pool_kernel_sizes[3]
        self.height_reduction = self.height_reduction * self.pool_kernel_sizes[3]

        ###print(x.shape)
        #x = x.view(x.shape[0], x.shape[3], -1)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 2, 3)
        x = torch.flatten(x, start_dim=2)
        ###print(x.shape)
        output, (hn, cn) = self.lstm(x)
        output = self.fully_connected(output)
        output = self.softmax(output)
        ##print("output")
        ##print(output)    
        return output
