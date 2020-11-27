import torch
import torch.nn as nn
import numpy as np

__all__ = ['CNN_MNIST', 'CNN_MNIST_Grad', 'LeNet5_TVGrad', 'LeNet5_TVGrad_logit', 'TVGradLoss', 'TKGradLoss']

class ConvGrad(nn.Module):
    # convolutional network with gradient computation
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, input_size=5):
        super(ConvGrad, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.padding = padding
        self.conv = nn.Conv2d(self.in_channels, self.out_channels,
                              self.kernel_size, padding=padding)

    def forward(self, x):
        y = x
        y = self.conv(x)

        return y

    def Gradient(self, g):
        G = nn.functional.conv2d(input=g, weight=torch.flip(torch.transpose(self.conv.weight,0,1), [2, 3]), padding=self.padding)
        return G

class AvgPoolGrad(nn.Module):
    def __init__(self, in_channels, device):
        super(AvgPoolGrad, self).__init__()
        self.num_channels = in_channels
        self.weight = (torch.zeros(self.num_channels, self.num_channels, 2, 2) + 0.25).to(device)

    def forward(self, x):
        return nn.functional.conv2d(input=x, weight=self.weight, stride=2)

    def Gradient(self, x):
        return nn.functional.conv_transpose2d(input=x, weight=self.weight, stride=2)

class MaxPoolGrad(nn.Module):
    def __init__(self):
        super(MaxPoolGrad, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.indices = torch.tensor([])

    def forward(self, x):
        y, self.indices = self.maxpool(x)
        return y

    def Gradient(self, x):
        return nn.functional.max_unpool2d(x, self.indices, 2)

class LinearGrad(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearGrad, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)

    def Gradient(self, x):
        y = nn.functional.linear(x, torch.transpose(self.fc.weight,0,1))
        return y

class SigmoidGrad(nn.Module):
    def __init__(self):
        super(SigmoidGrad, self).__init__()
        self.channels = torch.tensor([])

    def forward(self, x):
        # self.channels = x
        # e=torch.exp(x)
        # return e/(e+1)
        self.channels = torch.sigmoid(x)
        return self.channels

    def Gradient(self, x):
        # e = torch.exp(self.channels)
        # grad = (e)/torch.pow(e+1,2)
        # grad = grad * x
        # return grad

        return self.channels * (1. - self.channels) * x

class ReLUGrad(nn.Module):
    def __init__(self):
        super(ReLUGrad, self).__init__()
        self.channels = torch.tensor([])

    def forward(self, x):
        self.channels = nn.functional.relu(x)
        return self.channels

    def Gradient(self, x):
        # print('ReLUGrad: ', x.shape, self.channels.shape)
        return x * torch.sign(self.channels) 


class TahnGrad(nn.Module):
    def __init__(self):
        super(TahnGrad, self).__init__()
        self.channels = torch.tensor([])

    def forward(self, x):
        self.channels = x
        e=torch.exp(2*x)
        return (e-1)/(e+1)

    def Gradient(self, x):
        e = torch.exp(2*self.channels)
        grad = (4*e)/torch.pow(e+1,2)
        grad = grad * x
        return grad

class DropoutGrad(nn.Module):
    def __init__(self, prob=0.5, device='cpu'):
        super(DropoutGrad, self).__init__()
        self.channels = torch.tensor([])
        self.device = device
        self.prob = prob

    def forward(self, x, train):
        self.channels = nn.functional.dropout2d(torch.ones(x.size()), p=self.prob, training=train).to(self.device)
        return x * self.channels

    def Gradient(self, x):
        return self.channels * x

class SoftMax(nn.Module):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.channels = torch.tensor([])

    def forward(self, x):
        # y = torch.exp(x)
        # self.sum = torch.sum(y, dim=1).view(-1,1)
        # y = y / self.sum
        y = nn.Softmax(dim=1)(x)
        self.channels = y
        return y

    def Gradient(self, ind):
        g = -self.channels[:,ind:ind+1] * self.channels
        g[:,ind] = g[:,ind] + self.channels[:,ind]
        return g

class CopyGrad(nn.Module):
    def __init__(self):
        super(CopyGrad, self).__init__()
        self.channels = torch.tensor([])

    def forward(self, x):
        self.channels = x
        return x

    def Gradient(self, idx):
        g = torch.zeros_like(self.channels)
        g.data[:, idx] = 1.
        return g

class CNN_MNIST_DO(nn.Module):
    def __init__(self, in_channels=1, num_class=10, input_size=28, prob=0.5, device='cuda:0'):
        super(CNN_MNIST_DO, self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.input_size = input_size
        # self.train = True
        self.prob = prob
        self.device = device
        self.num_channels=[16, 32]
        self.conv1 = ConvGrad(in_channels=self.in_channels, out_channels=self.num_channels[0], kernel_size=3, input_size=self.input_size)
        self.activation1 = SigmoidGrad()
        self.dropout1 = DropoutGrad(device=device)
        self.pool1 = MaxPoolGrad()
        self.conv2 = ConvGrad(in_channels=self.num_channels[0], out_channels=self.num_channels[1], kernel_size=3, input_size=self.input_size)
        self.activation2 = SigmoidGrad()
        self.dropout2 = DropoutGrad(device=device)
        self.pool2 = MaxPoolGrad()
        self.fc1 = LinearGrad(self.num_channels[1]*int(input_size/4)**2, self.num_class)
        self.softmax = SoftMax()

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.activation1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.activation2(y)
        y = self.dropout2(y)
        y = self.fc1(y.view(-1, int(self.input_size/4)**2*self.num_channels[1]))
        y = self.softmax(y)
        return y

class CNN_MNIST_Grad(nn.Module):
    def __init__(self, in_channels=1, num_class=10, input_size=28, dropout_rate=0., alpha = .05, device='cpu'):
        super(CNN_MNIST_Grad, self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.input_size = input_size
        self.training = True
        self.prob = dropout_rate
        self.device = device
        self.num_channels=[16, 64]
        self.conv1 = ConvGrad(in_channels=self.in_channels, out_channels=self.num_channels[0], kernel_size=3, input_size=self.input_size)
        # self.activation1 = SigmoidGrad()
        self.activation1 = ReLUGrad()
        self.dropout1 = DropoutGrad(self.prob, device=device)
        self.pool1 = MaxPoolGrad()
        self.conv2 = ConvGrad(in_channels=self.num_channels[0], out_channels=self.num_channels[1], kernel_size=3, input_size=self.input_size)
        # self.activation2 = SigmoidGrad()
        self.activation2 = ReLUGrad()
        self.dropout2 = DropoutGrad(self.prob, device=device)
        self.pool2 = MaxPoolGrad()
        self.fc1 = LinearGrad(self.num_channels[1]*int(input_size/4)**2, 1000)
        # self.activation3 = SigmoidGrad()
        self.activation3 = ReLUGrad()
        self.fc2 = LinearGrad(1000, self.num_class)
        self.softmax = SoftMax()
        self.grad = None

        self.alpha = alpha

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.activation1(y)
        if self.prob>0:
            y = self.dropout1(y, self.training)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.activation2(y)
        if self.prob>0:
            y = self.dropout2(y, self.training)
        y = self.fc1(y.view(-1, int(self.input_size/4)**2*self.num_channels[1]))
        y = self.activation3(y)
        logit = self.fc2(y)
        

        if self.training:
            y = self.softmax(logit)
            self.grad = self.Gradient()
            return logit, self.grad
        else:
            return logit

    def Gradient(self):
        grads = []
        for i in range(self.num_class):
            grad = self.softmax.Gradient(i)
            # grad = torch.mm(g.view(-1,self.num_class), self.fc1.weight)
            grad = self.fc2.Gradient(grad)
            grad = self.activation3.Gradient(grad)
            grad = self.fc1.Gradient(grad)
            grad = grad.view(-1, self.num_channels[1], int(self.input_size/4), int(self.input_size/4))
            if self.prob>0:
                grad = self.dropout2.Gradient(grad)
            grad = self.activation2.Gradient(grad)
            grad = self.pool2.Gradient(grad)
            grad = self.conv2.Gradient(grad)
            if self.prob>0:
                grad = self.dropout1.Gradient(grad)
            grad = self.activation1.Gradient(grad)
            grad = self.pool1.Gradient(grad)
            grad = self.conv1.Gradient(grad)

            grad = grad.view(grad.shape[0], -1)
            grads.append(grad * self.alpha)

        return grads


class LeNet5_TVGrad(nn.Module):
    def __init__(self):
        super(LeNet5_TVGrad, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv1 = ConvGrad(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.pool1 = MaxPoolGrad()
        self.activation1 = ReLUGrad()

        # self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.conv2 = ConvGrad(32, 64, 5, padding=2)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.pool2 = MaxPoolGrad()
        self.activation2 = ReLUGrad()
        
        # self.fc1 = nn.Linear(64*49, 1024)
        # self.fc2 = nn.Linear(1024, 10)
        self.fc1 = LinearGrad(64*49, 1024)
        self.fc2 = LinearGrad(1024, 10)
        self.activation3 = ReLUGrad()

        self.softmax = SoftMax()
        
        self.grad = None
        # self.activation = nn.ReLU()

    def forward(self, x):
        # 1x28x28 -> 32x14x14
        x = self.activation1(self.pool1(self.conv1(x)))
        
        # 32x14x14 -> 64x7x7
        x = self.activation2(self.pool2(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64*49)
        
        # FC1: 64*49 -> 1024
        x = self.activation3(self.fc1(x))
        
        # FC2: 1024 -> 10
        logit = self.fc2(x)

        if self.training:
            y = self.SoftMax(logit)
            self.grad = self.Gradient()
            return logit, self.grad
        else:
            return logit

    def Gradient(self):
        grads = []
        for i in range(10):
            grad = self.softmax.Gradient(i)
            # grad = torch.mm(g.view(-1,self.num_class), self.fc1.weight)
            grad = self.fc2.Gradient(grad)
            grad = self.activation3.Gradient(grad)
            grad = self.fc1.Gradient(grad)
            grad = grad.view(-1, 64, 7, 7)
            grad = self.activation2.Gradient(grad)
            grad = self.pool2.Gradient(grad)
            grad = self.conv2.Gradient(grad)
            grad = self.activation1.Gradient(grad)
            grad = self.pool1.Gradient(grad)
            grad = self.conv1.Gradient(grad)

            grad = grad.view(grad.shape[0], -1)
            grads.append(grad)

        return grads

class LeNet5_TVGrad_logit(nn.Module):
    def __init__(self):
        super(LeNet5_TVGrad_logit, self).__init__()
        # self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv1 = ConvGrad(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        # self.pool1 = nn.MaxPool2d(2, 2)
        self.pool1 = MaxPoolGrad()
        self.activation1 = ReLUGrad()

        # self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.conv2 = ConvGrad(32, 64, 5, padding=2)
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.pool2 = MaxPoolGrad()
        self.activation2 = ReLUGrad()
        
        # self.fc1 = nn.Linear(64*49, 1024)
        # self.fc2 = nn.Linear(1024, 10)
        self.fc1 = LinearGrad(64*49, 1024)
        self.fc2 = LinearGrad(1024, 10)
        self.activation3 = ReLUGrad()

        self.softmax = SoftMax()
        self.last_layer = CopyGrad()
        
        self.grad = None
        # self.activation = nn.ReLU()

    def forward(self, x):
        # 1x28x28 -> 32x14x14
        x = self.activation1(self.pool1(self.conv1(x)))
        
        # 32x14x14 -> 64x7x7
        x = self.activation2(self.pool2(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64*49)
        
        # FC1: 64*49 -> 1024
        x = self.activation3(self.fc1(x))
        
        # FC2: 1024 -> 10
        logit = self.fc2(x)

        if self.training:
            y = self.last_layer(logit)
            self.grad = self.Gradient()
            return logit, self.grad
        else:
            return logit

    def Gradient(self):
        grads = []
        for i in range(10):
            grad = self.last_layer.Gradient(i)
            # grad = torch.mm(g.view(-1,self.num_class), self.fc1.weight)
            grad = self.fc2.Gradient(grad)
            grad = self.activation3.Gradient(grad)
            grad = self.fc1.Gradient(grad)
            grad = grad.view(-1, 64, 7, 7)
            grad = self.activation2.Gradient(grad)
            grad = self.pool2.Gradient(grad)
            grad = self.conv2.Gradient(grad)
            grad = self.activation1.Gradient(grad)
            grad = self.pool1.Gradient(grad)
            grad = self.conv1.Gradient(grad)

            grad = grad.view(grad.shape[0], -1)
            grads.append(grad)

        return grads


class CNN_MNIST(nn.Module):
    def __init__(self, in_channels=1, num_class=10, input_size=28, dropout_rate=0., device='cpu'):
        super(CNN_MNIST, self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.input_size = input_size
        self.training = True
        self.prob = dropout_rate
        self.device = device
        self.num_channels=[16, 64]
        self.conv1 = ConvGrad(in_channels=self.in_channels, out_channels=self.num_channels[0], kernel_size=3, input_size=self.input_size)
        # self.activation1 = SigmoidGrad()
        self.activation1 = ReLUGrad()
        self.dropout1 = DropoutGrad(self.prob, device=device)
        self.pool1 = MaxPoolGrad()
        self.conv2 = ConvGrad(in_channels=self.num_channels[0], out_channels=self.num_channels[1], kernel_size=3, input_size=self.input_size)
        # self.activation2 = SigmoidGrad()
        self.activation2 = ReLUGrad()
        self.dropout2 = DropoutGrad(self.prob, device=device)
        self.pool2 = MaxPoolGrad()
        self.fc1 = LinearGrad(self.num_channels[1]*int(input_size/4)**2, 1000)
        # self.activation3 = SigmoidGrad()
        self.activation3 = ReLUGrad()
        self.fc2 = LinearGrad(1000, self.num_class)
        self.softmax = SoftMax()

    def forward(self, x):
        y = self.conv1(x)
        y = self.pool1(y)
        y = self.activation1(y)
        if self.prob>0:
            y = self.dropout1(y, self.training)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.activation2(y)
        if self.prob>0:
            y = self.dropout2(y, self.training)
        y = self.fc1(y.view(-1, int(self.input_size/4)**2*self.num_channels[1]))
        y = self.activation3(y)
        y = self.fc2(y)
        # y = self.softmax(y)
        return y


class TVGradLoss(nn.Module):
    def __init__(self, alpha=5.):
        super(TVGradLoss, self).__init__()
        self.alpha = alpha

    def forward(self, data, target):
        logit, grads = data[0], data[1]
        Loss = nn.CrossEntropyLoss()(logit, target)

        if isinstance(grads, list) or isinstance(grads, tuple):
            for grad in grads:
                Reg = torch.norm(grad, p=2, dim=1).mean() * self.alpha # torch.norm
                Loss = Loss + Reg
        else:
            Reg = torch.norm(grad, p=2, dim=1).mean() * self.alpha
            Loss += Reg

        return Loss 

class TKGradLoss(nn.Module):
    def __init__(self, alpha=5.):
        super(TKGradLoss, self).__init__()
        self.alpha = alpha

    def forward(self, data, target):
        logit, grads = data[0], data[1]
        Loss = nn.CrossEntropyLoss()(logit, target)

        if isinstance(grads, list) or isinstance(grads, tuple):
            for grad in grads:
                Reg = nn.MSELoss()(grad, torch.zeros_like(grad)) * self.alpha # torch.norm
                Loss = Loss + Reg
        else:
            Reg = nn.MSELoss()(grad, torch.zeros_like(grad)) * self.alpha
            Loss += Reg

        return Loss 

def main():
    batch_size = 1
    input_channels = 1
    input_size = 4
    num_class = 2

    x = torch.rand(batch_size, input_channels, input_size, input_size)
    x.requires_grad=True
    model = CNN_MNIST_Grad(in_channels=input_channels, num_class=num_class, input_size=input_size)
    
    model.eval()
    y = nn.Softmax(dim=1)(model(x))
    G = model.Gradient()
    # y[0,1].backward()
    z = torch.sum(y, dim=0)
    print(z.shape)
    z[1].backward()
    print(x.grad)
    print(G[1])


if __name__ == '__main__':
    main()

