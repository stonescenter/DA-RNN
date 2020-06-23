
import os

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from livelossplot import PlotLosses



# https://www.programcreek.com/python/example/104440/torch.nn.functional.nll_loss

# change if you will use in local machine
#writer = SummaryWriter(log_dir='/content/cloned-repo/runs')

class Autoencoder(nn.Module):
    def __init__(self, device, epochs, original_dim, intermediate_dim):

        super(Autoencoder, self).__init__()
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.num_epochs = epochs

        self.encoder = nn.Sequential(
            nn.Linear(original_dim, 20),
            nn.ReLU(True),
            nn.Linear(20, 15),
            nn.ReLU(True), 
            nn.Linear(15, 12),
            nn.ReLU(True),
            nn.Linear(12, intermediate_dim))
        
        self.decoder = nn.Sequential(
            nn.Linear(intermediate_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 15),
            nn.ReLU(True),
            nn.Linear(15, 20),
            nn.ReLU(True), 
            nn.Linear(20, original_dim),
            nn.Tanh())

        learning_rate = 1e-3
        #self.criterion = nn.MSELoss()
        self.criterion = nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.parameters(),
            lr=learning_rate, weight_decay=1e-5)
        
        self.device = device

    def accuracy(model, data_x, data_y, pct_close):
        # data_x and data_y are numpy array-of-arrays matrices
        n_feat = len(data_x[0])  # number features
        n_items = len(data_x)    # number items
        n_correct = 0; n_wrong = 0
        for i in range(n_items):
            X = T.Tensor(data_x[i])
            # Y = T.Tensor(data_y[i])  # not needed
            oupt = model(X)            # Tensor
            pred_y = oupt.item()       # scalar

            if np.abs(pred_y - data_y[i]) < \
                np.abs(pct_close * data_y[i]):
                n_correct += 1
            else:
                n_wrong += 1
        return (n_correct * 100.0) / (n_correct + n_wrong)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def train_model(self, train_loader):

        for epoch in range(self.num_epochs):
            logs = {}
            for inputs, labels in train_loader:

                inputs = torch.DoubleTensor(inputs).to(self.device)
                #self.optimizer.zero_grad()
                inputs = Variable(inputs).to(self.device)

                #self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, inputs)
                
                #for train
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('[Model] epoch=%s, loss=%s' % ( epoch, loss.item()))
            #writer.add_scalar('Train/Loss', loss.item(), epoch)

    def train_advanced(self, data_loaders, show_plot=True):
        liveloss = PlotLosses()

        for epoch in range(self.num_epochs):
            logs = {}
        
            for phase in ['train', 'validation']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in data_loaders[phase]:

                    inputs = torch.DoubleTensor(inputs).to(self.device)
                    targets = torch.DoubleTensor(inputs).to(self.device)

                    #inputs = torch.DoubleTensor(inputs)
                    inputs = Variable(inputs).to(self.device)
                    targets = Variable(targets).to(self.device)

                    #self.optimizer.zero_grad()
                    #outputs = self.forward(inputs)
                    outputs = self.encoder(inputs)
                    outputs = self.decoder(outputs)
                    loss = self.criterion(outputs, inputs)
                    
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    _, preds = torch.max(outputs, 1)

                    running_loss += loss.detach() * inputs.size(0)
                    #running_corrects += torch.sum(preds == inputs.data)
                
                epoch_loss = running_loss / len(data_loaders[phase].dataset)
                #epoch_acc = running_corrects.float() / len(data_loaders[phase].dataset)

                prefix = ''
                if phase == 'validation':
                    prefix = 'val_'

                    #print('[Model] epoch=%s, loss=%s , acc=%s' % ( epoch, loss.item(), epoch_acc.item))
                    print('[Model] epoch=%s, loss1=%s, loss2=%s ' % ( epoch, loss.item(), epoch_loss.item()))

                logs[prefix + 'log loss'] = epoch_loss.item()
                #logs[prefix + 'accuracy'] = epoch_acc.item()
                
                if show_plot:
                    liveloss.update(logs)
                    liveloss.send()

    # def test(self, test_loader):       

    #     self.eval()
    #     test_loss = 0
    #     correct = 0
        
    #     for inputs, labels in test_loader:

    #         inputs = torch.DoubleTensor(inputs).to(self.device)
    #         inputs = Variable(inputs).to(self.device)

    #         output = self(inputs)
    #         #output = output.detach()
    #         target = Variable(torch.FloatTensor(output))
    #         test_loss += F.nll_loss(output, target.view(len(target),-1), size_average=False).data[0] # sum up batch loss
    #         pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability          
    #         correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    #     test_loss /= len(test_loader.dataset)

    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset)))        
    #     print("debug")

    def test(self, test_loader):

        self.eval()

        predicted = []
        for inputs, labels in test_loader:

            #inputs = inputs.to(device)
            inputs = inputs.type(torch.DoubleTensor)
            target = inputs.type(torch.FloatTensor)

            if str(self.device) == str('cuda'):
                inputs, target = inputs.to(self.device), inputs.to(self.device)
            
            #torch.DoubleTensor(inputs)
            target = Variable(target)
            inputs = Variable(inputs)
            outputs = self.encoder(inputs) # devuelve com grd_fn
            #outputs = outputs.detach()

            # negative log likehood loss
            #print(outputs.size())
            outputs = outputs.view(len(outputs), -1)
            target = target.view(len(target),-1)

            outputs = outputs.detach().cpu().numpy()
            
            predicted.append(outputs)

        #return predicted.ravel()
        results = []
        for i in range(len(predicted)):
            x = predicted[i]
            for j in range(len(x)):
                d = x[j]
                results.append(d)
        
        return results

    def test_advanced(self, test_loader):
        
        predicted = []

        dtype = torch.float
        device = torch.device(self.device)

        for inputs, labels in test_loader:

            #inputs = inputs.to(device)
            inputs = inputs.type(torch.DoubleTensor)
            target = inputs.type(torch.FloatTensor)

            if str(self.device) == str('cuda'):
                inputs, target = inputs.to(self.device), inputs.to(self.device)
            
            #torch.DoubleTensor(inputs)
            target = Variable(target)
            inputs = Variable(inputs)
            outputs = self.encoder(inputs) # devuelve com grd_fn
            #outputs = outputs.detach()

            # negative log likehood loss
            outputs = outputs.view(len(outputs), outputs.size(2))
            target = target.view(len(target),-1)

            outputs = outputs.detach().cpu().numpy()
            
            predicted.append(outputs)

        #return predicted.ravel()
        results = []
        for i in range(len(predicted)):
            x = predicted[i]
            for j in range(len(x)):
                d = x[j]
                results.append(d)
        
        return results
    def save_model(self, filepath):
        checkpoint = {'model': self,
                'state_dict': self.state_dict()}
                #'encoder_optimizer' : self.encoder_optimizer.state_dict(),
                #'decoder_optimizer' : self.decoder_optimizer.state_dict()}
        
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        #for parameter in model.parameters():
        #    parameter.requires_grad = False

        # colocamos em modo de trainamento falso
        #model.eval()
        #model.train(False)
        return model

class Autoencoder2(nn.Module):
    def __init__(self):
        super(Autoencoder2,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x    