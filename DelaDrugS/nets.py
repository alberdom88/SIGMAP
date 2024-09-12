import torch.nn as nn
import numpy as np
import torch    
 

class MyRnnNet2Layers(nn.Module):
    def __init__(self, params):
        super(MyRnnNet2Layers, self).__init__()
        self.inputSize = params["inputSize"]
        self.hiddenSize_L1 = params["hL1"]
        self.hiddenSize_L2 = params["hL2"]
        self.hiddenSize_L3 = params["hLinear"]
        self.outSize = params["outputSize"]
        
        self.l2 = nn.LSTMCell(self.inputSize, self.hiddenSize_L1)
        self.l3 = nn.LSTMCell(self.hiddenSize_L1, self.hiddenSize_L2)
        self.l4 = nn.Linear(self.hiddenSize_L2, self.hiddenSize_L3)
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(self.hiddenSize_L3, self.outSize)  
    
    def forward(self, x, hx1, cx1, hx2, cx2):
        hx1, cx1 = self.l2(x, (hx1, cx1))
        hx2, cx2 = self.l3(hx1, (hx2, cx2))
        out = self.l4(hx2)
        out = self.relu(out)
        out = self.l5(out)
        # no activation and no softmax at the end
        return out, hx1, cx1, hx2, cx2

class MyRnnNet2LayersBi(nn.Module):
    def __init__(self, params):
        super(MyRnnNet2LayersBi, self).__init__()
        self.inputSize = params["inputSize"]
        self.hiddenSize_L1 = params["hL1"]
        self.hiddenSize_L2 = params["hL2"]
        self.hiddenSize_L3 = params["hLinear"]
        self.outSize = params["outputSize"]
        
        self.l2 = nn.LSTMCell(self.inputSize, self.hiddenSize_L1)
        self.l2b = nn.LSTMCell(self.inputSize, self.hiddenSize_L1)
        self.l3 = nn.LSTMCell(self.hiddenSize_L1 * 2, self.hiddenSize_L2)
        self.l4 = nn.Linear(self.hiddenSize_L2, self.hiddenSize_L3)
        self.relu = nn.ReLU()
        self.l5 = nn.Linear(self.hiddenSize_L3, self.outSize)  
    
    def forward(self, x, hx1, cx1, hx1b, cx1b, hx2, cx2):
        hx1, cx1 = self.l2(x, (hx1, cx1))
        hx1b, cx1b = self.l2b(torch.flip(x,(1,)), (hx1b, cx1b))
        hx2, cx2 = self.l3(torch.cat((hx1,hx1b),1), (hx2, cx2))
        out = self.l4(hx2)
        out = self.relu(out)
        out = self.l5(out)
        # no activation and no softmax at the end
        return out, hx1, cx1, hx1b, cx1b, hx2, cx2    
    
class MyRnnNet3Layers(nn.Module):
    def __init__(self, params):
        super(MyRnnNet3Layers, self).__init__()
        self.inputSize = params["inputSize"]
        self.embeddingSize = params["embeddingSize"]
        self.hiddenSize_L1 = params["hL1"]
        self.hiddenSize_L2 = params["hL2"]
        self.hiddenSize_L3 = params["hL3"]
        self.hiddenSize_L4 = params["hLinear"]
        self.outSize = params["outputSize"]

        self.l1 = nn.Linear(self.inputSize, self.embeddingSize) 
        self.l2 = nn.LSTMCell(self.embeddingSize, self.hiddenSize_L1)
        self.l3 = nn.LSTMCell(self.hiddenSize_L1, self.hiddenSize_L2)
        self.l4 = nn.LSTMCell(self.hiddenSize_L2,  self.hiddenSize_L3)
        self.l5 = nn.Linear(self.hiddenSize_L3, self.hiddenSize_L4)
        self.relu = nn.ReLU()
        self.l6 = nn.Linear(self.hiddenSize_L4, self.outSize)  

    def forward(self, x, hx1, cx1, hx2, cx2,hx3,cx3):
        out = self.l1(x)
        hx1, cx1 = self.l2(out, (hx1, cx1))
        hx2, cx2 = self.l3(hx1, (hx2, cx2))
        hx3, cx3 = self.l4(hx2, (hx3, cx3))
        out = self.l5(hx3)
        out = self.relu(out)
        out = self.l6(out)
        # no activation and no softmax at the end
        return out, hx1, cx1, hx2, cx2, hx3, cx3
