from __future__ import print_function, division, absolute_import, unicode_literals
import six

import os
import librosa
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,savefig
import copy
from sys import argv

class CNNModel(nn.Module):
    def __init__(self,blockSize):
        super(CNNModel,self).__init__()
        self.blockSize = blockSize
        self.cnn = nn.Conv1d(in_channels=1025,out_channels=4096,kernel_size=3,stride=1,padding=1)
        self.activate = nn.SELU(True)

        #Set the random parameters to be constant.
        weight =torch.randn(self.cnn.weight.data.shape)
        self.cnn.weight = torch.nn.Parameter(weight,requires_grad=False)
        bias = torch.zeros(self.cnn.bias.data.shape)
        self.cnn.bias = torch.nn.Parameter(bias,requires_grad=False)

    
    #def initParams(self):
        #for param in self.parameters():
            #if len(param.shape)>1:
                #nn.init.xavier_normal_(param)
        
        
    def forward(self,x):
        out = self.activate(self.cnn(x))
        out = out.view(out.size(0),-1)
        return out


def GramMatrix(input):
    batch_size,Num_feature,n_Frame = input.size()
    features = input.view(batch_size*Num_feature,n_Frame)
    G = torch.mm(features,features.t())
    G_normal = G.div(n_Frame)
    return G_normal


class StyleLoss(nn.Module):
    def __init__(self,target_feature):
        super(StyleLoss,self).__init__()
        self.target = GramMatrix(target_feature).detach()
        #self.weight = weight
        #self.gram = GramMatrix()
        #self.Loss = nn.MSELoss()
        
    def forward(self,input):
        #self.output = input.clone()
        G = GramMatrix(input)
        #self.G.mul_(self.weight)
        self.loss = F.mse_loss(G,self.target)
        return input
    
    #def backward(self,retain_graph=True):
        #self.loss.backward(retain_graph=retain_graph)
        #return self.loss


class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        #self.weight = weight
        #self.Loss = nn.MSELoss()

    def forward(self, input):
        #self.output = input.clone()
        self.loss = F.mse_loss(input, self.target)
        return input

    #def backward(self,retain_graph=True):
        #self.loss.backward(retain_graph=retain_graph)
        #return self.loss


if __name__ == '__main__':
    N_FFT = 2048
    blockSize = 1024

    #print('Enter the names of SCRIPT,CONTENT,STYLE')
    script,content_name,style_name = argv
    def read_audio_spectum(filename):
        x, fs = librosa.load(filename)
        S = librosa.stft(x,n_fft=N_FFT,win_length=blockSize)
        S = np.log1p(np.abs(S))
        return S,fs
    
    style_audio_np, style_fs = read_audio_spectum(style_name)
    content_audio_np, content_fs = read_audio_spectum(content_name)
    nFrame = style_audio_np.shape[1]
    style_audio_np = style_audio_np.reshape([1,1025,nFrame])
    content_audio_np = content_audio_np.reshape([1,1025,-1])

    style_audio = torch.from_numpy(style_audio_np)
    content_audio = torch.from_numpy(content_audio_np)

    

    if (content_fs == style_fs):
        print('Sampling Rates are same')
    
    else:
        print('Sampling Rates are different')
        exit()

    cnnmodel = CNNModel(blockSize)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    cnnmodel.to(device)
    style_weight = 1
    content_weight = 1e-2
    style_layers_default = ['conv_1']

    def style_model_and_loss(cnnmodel,style_audio,style_weight=style_weight,content_weight=content_weight,style_layers=style_layers_default):

        cnnmodel = copy.deepcopy(cnnmodel)
        style_losses = []
        content_losses = []

        model_style = nn.Sequential()
        #gram = GramMatrix() #for style target computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_style.to(device)
        #gram.to(device)

        name = 'conv_1'
        name_activate = 'selu_1'
        model_style.add_module(name,cnnmodel.cnn)
        model_style.add_module(name_activate,cnnmodel.activate)
        if name in style_layers:#only one layer so content layer and style layer are the same
            target = model_style(content_audio).detach()
            target_feature = model_style(style_audio).detach()
            style_loss= StyleLoss(target_feature)
            content_loss=ContentLoss(target)
            model_style.add_module("style_loss_1",style_loss)
            model_style.add_module("content_loss_1",content_loss)
            style_losses.append(style_loss)
            content_losses.append(content_loss)
        
        return model_style,style_losses,content_losses

    input_audio = content_audio.clone()
    #input_audio = Variable(torch.randn(content_audio.shape) * 1e-3, requires_grad=True)

    learning_rate_initial = 0.03

    #def input_param_opt(input_audio):#we just wanna optimize input,but not model
     #   input_param = nn.Parameter(input_audio.data)
     #   optimizer = optim.Adam([input_param],lr=learning_rate_initial,betas=(0.9,0.999),eps=1e-8)
      #  return input_param,optimizer

    def input_param_opt(input_audio):
    # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_audio.requires_grad_()])
        input_param = nn.Parameter(input_audio.data)
        return input_param,optimizer

    num_steps = 300


    def run_style_transfer(cnnmodel,style_audio,content_audio,input_audio,num_steps=num_steps,style_weight=style_weight,content_weight=content_weight):
        print('Building the style transfer model...')
        model,style_losses,content_losses = style_model_and_loss(cnnmodel,style_audio,style_weight,content_weight)
        print(model)
        input_param,optimizer = input_param_opt(input_audio)
        print(input_param)
        print('Opt...')
        run_step = [0]
        print(num_steps)
        print(style_losses)
        print(content_losses)

        while run_step[0] <= num_steps:
            print(run_step[0])
            def closure():
            	

                optimizer.zero_grad()
                model(input_audio)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    #print('sl is ',sl,' style loss is ',sl.loss)
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()

                run_step[0] += 1
                if run_step[0] % 10 == 0:
                    print("run_step {}:".format(run_step))
                    print('Style Loss : {:8f}'.format(style_score.data[0])) #CHANGE 4->8 
                    print('Content Loss:{:8f}'.format(content_score.data[0]))
                    

                return style_score + content_score
            optimizer.step(closure)
            

            

            
            


		# 	optimizer.step(closure)
		# input_param.data.clamp_(0, 1)
        #     #style_loss_sum = 0
        #     optimizer.zero_grad()
        #     model(input_param)
        #     for sl in style_losses:
        #         style_loss_sum += sl.backward()
        #     print(style_loss_sum)
        #     optimizer.step()
        #     run_step+=1
        return input_param.data

        
    output = run_style_transfer(cnnmodel,style_audio,content_audio, input_audio)

    if torch.cuda.is_available():
        output = output.cpu()

    output = output.squeeze(0)
    output = output.numpy()
    print(output.shape)

    ang = np.zeros_like(output)
    ang = np.exp(output) - 1

    #phase reconstruction
    p = 2 *np.pi *np.random.random_sample(ang.shape) - np.pi
    for i in range(500):
        S = ang*np.exp(1j*p)
        x= librosa.istft(S,win_length=blockSize)
        p = np.angle(librosa.stft(x,n_fft=N_FFT,win_length=blockSize))

    OUTPUT_FILENAME =  'output1D_4096_iter'+str(num_steps)+'_c'+content_name+'_s'+style_name+'_sw'+str(style_weight)+'_k3s1p1.wav'
    librosa.output.write_wav(OUTPUT_FILENAME,x,style_fs)

    print('DONE...')

