import torch
from torch.nn import Linear, ReLU, Sigmoid, Dropout, BatchNorm1d

class MLP(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 device: str,
                 classes={},
                 **kwargs):

        super(MLP,self).__init__()

        self.classes = classes
        self.num_classes = len(self.classes)
        self.input_channels = input_channels

        # First hidden layer
        self.hidden1 = Linear(self.input_channels, 100)
        self.bn1 = BatchNorm1d(100)
        self.act1 = ReLU()

        # Second hidden layer
        self.do2 = Dropout(0.25)
        self.hidden2 = Linear(100, 500)
        self.bn2 = BatchNorm1d(500)
        self.act2 = ReLU()

        # Third hidden layer
        self.do3 = Dropout(0.25)
        self.hidden3 = Linear(500, 250)
        self.bn3 = BatchNorm1d(250)
        self.act3 = ReLU()

        # Fourth hidden layer
        self.do4 = Dropout(0.25)
        self.hidden4 = Linear(250, 100)
        self.bn4 = BatchNorm1d(100)
        self.act4 = ReLU()

        # Fifth hidden layer
        self.do5 = Dropout(0.25)
        self.hidden5 = Linear(100,25)
        self.bn5 = BatchNorm1d(25)
        self.act5 = ReLU()

        # Sixth hidden layer
        self.hidden6 = Linear(25,1)
        self.act6 = Sigmoid()

        self.device = device
        self.to(device)

    def forward(self, X):

        #Input to the first hidden layer
        X = self.hidden1(X)
        X = self.bn1(X.permute(0,2,1))
        X = self.act1(X.permute(0,2,1))

        # Second hidden layer
        X = self.do2(X)
        X = self.hidden2(X)
        X = self.bn2(X.permute(0,2,1))
        X = self.act2(X.permute(0,2,1))

        # Third hidden layer
        X = self.do3(X)
        X = self.hidden3(X)
        X = self.bn3(X.permute(0,2,1))
        X = self.act3(X.permute(0,2,1))

        # Fourth hidden layer
        X = self.do4(X)
        X = self.hidden4(X)
        X = self.bn4(X.permute(0,2,1))
        X = self.act4(X.permute(0,2,1))

        # Fifth hidden layer
        X = self.do5(X)
        X = self.hidden5(X)
        X = self.bn5(X.permute(0,2,1))
        X = self.act5(X.permute(0,2,1))

        # Sixth hidden layer
        X = self.hidden6(X)
        X = self.act6(X)
        
        return X