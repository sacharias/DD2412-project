import copy
import torch

class EMA():
  def __init__(self, decay, model):
      self.decay = decay
      self.emamodel = copy.deepcopy(model)
      self.emamodel.cuda()
  
  def update(self, model):
    state_dict_emamodel = self.emamodel.state_dict()
    #Jag är paranoid att gradienten kanske flödar igenom?
    #torch.no_grad kanske gör det snabbare
    with torch.no_grad():
      for name, param in model.state_dict().items():
        state_dict_emamodel[name] = self.decay * state_dict_emamodel[name]  + (1 - self.decay) * param
