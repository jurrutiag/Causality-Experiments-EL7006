import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from models3 import *

from causal_meta.utils.torch_utils import logsumexp

class BinaryStructuralModel(nn.Module):
    def __init__(self, model_A_B, model_B_A):
        super(BinaryStructuralModel, self).__init__()
        self.model_A_B = model_A_B
        self.model_B_A = model_B_A
        self.w = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, inputs):
        return self.online_loglikelihood(self.model_A_B(inputs), self.model_B_A(inputs))

    def online_loglikelihood(self, logl_A_B, logl_B_A):
        n = logl_A_B.size(0)
        log_alpha, log_1_m_alpha = F.logsigmoid(self.w), F.logsigmoid(-self.w)

        return logsumexp(log_alpha + torch.sum(logl_A_B),
            log_1_m_alpha + torch.sum(logl_B_A))# / float(n)

    def modules_parameters(self):
        return chain(self.model_A_B.parameters(), self.model_B_A.parameters())

    def structural_parameters(self):
        return [self.w]
    
    
    
class BinaryStructuralModel_extended(nn.Module):
    def __init__(self, ModelA2BC, ModelB2AC, ModelC2AB, ModelA2BC_B2C, ModelA2BC_C2B, ModelB2AC_A2C, ModelB2AC_C2A, ModelC2AB_A2B, ModelC2AB_B2A):
        super(BinaryStructuralModel_extended, self).__init__()
        self.model_1 = ModelA2BC
        self.model_2 = ModelB2AC
        self.model_3 = ModelC2AB
        self.model_4 = ModelA2BC_B2C
        self.model_5 = ModelA2BC_C2B
        self.model_6 = ModelB2AC_A2C
        self.model_7 = ModelB2AC_C2A
        self.model_8 = ModelC2AB_A2B
        self.model_9 = ModelC2AB_B2A
        
        #Structural parameters:
        self.w1 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w2 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w3 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w4 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w5 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w6 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w7 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w8 = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.w9 = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, inputs):
            return self.online_loglikelihood(self.model_1(inputs), self.model_2(inputs), self.model_3(inputs), self.model_4(inputs), self.model_5(inputs), self.model_6(inputs),\
                                                        self.model_7(inputs), self.model_8(inputs), self.model_9(inputs))

    def online_loglikelihood(self, logl_1, logl_2, logl_3, logl_4, logl_5, logl_6, logl_7, logl_8, logl_9):
        structural = torch.cat([self.w1.reshape(1, -1), self.w2.reshape(1, -1), self.w3.reshape(1, -1), self.w4.reshape(1, -1), self.w5.reshape(1, -1), self.w6.reshape(1, -1), self.w7.reshape(1, -1), self.w8.reshape(1, -1), self.w9.reshape(1, -1)], dim=0)

        smax = F.softmax(structural, dim=0)
        pw1, pw2, pw3, pw4, pw5, pw6, pw7, pw8, pw9 = smax[0, 0], smax[1, 0], smax[2, 0], smax[3, 0], smax[4, 0], smax[5, 0], smax[6, 0], smax[7, 0], smax[8, 0]

        return -torch.log(pw1*torch.exp(torch.sum(logl_1)) + pw2*torch.exp(torch.sum(logl_2))  + pw3*torch.exp(torch.sum(logl_3)) + pw4*torch.exp(torch.sum(logl_4)) + pw5*torch.exp(torch.sum(logl_5)) + pw6*torch.exp(torch.sum(logl_6)) + pw7*torch.exp(torch.sum(logl_7)) + pw8*torch.exp(torch.sum(logl_8)) + pw9*torch.exp(torch.sum(logl_9)))
                

    def modules_parameters(self):
        return chain(self.model_1.parameters(), self.model_2.parameters(), self.model_3.parameters(), self.model_4.parameters(),\
                    self.model_5.parameters(), self.model_6.parameters(),  self.model_7.parameters(),\
                    self.model_8.parameters(), self.model_9.parameters())

    def structural_parameters(self):
        return [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7, self.w8, self.w9]

    def set_ground_truth(self, model, *args):
        self.model_1.set_ground_truth(model, *args)
        self.model_2.set_ground_truth(model, *args)
        self.model_3.set_ground_truth(model, *args)
        self.model_4.set_ground_truth(model, *args)
        self.model_5.set_ground_truth(model, *args)
        self.model_6.set_ground_truth(model, *args)
        self.model_7.set_ground_truth(model, *args)
        self.model_8.set_ground_truth(model, *args)
        self.model_9.set_ground_truth(model, *args)
    

class ModelA2B(nn.Module):
    def __init__(self, marginal, conditional):
        super(ModelA2B, self).__init__()
        self.p_A = marginal
        self.p_B_A = conditional

    def forward(self, inputs):
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        return self.p_A(inputs_A) + self.p_B_A(inputs_B, inputs_A)

class ModelB2A(nn.Module):
    def __init__(self, marginal, conditional):
        super(ModelB2A, self).__init__()
        self.p_B = marginal
        self.p_A_B = conditional

    def forward(self, inputs):
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        return self.p_B(inputs_B) + self.p_A_B(inputs_A, inputs_B)
