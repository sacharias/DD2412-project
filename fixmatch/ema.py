import copy
import torch

class EMA():
  def __init__(self, decay, model):
      self.decay = decay
      self.emamodel = copy.deepcopy(model)
      self.emamodel.cuda()
  
  def update(self, model):
    #Jag är paranoid att gradienten kanske flödar igenom?
    #torch.no_grad kanske gör det snabbare
    model_state_dict = model.state_dict()

    with torch.no_grad():
      for name, param in self.emamodel.state_dict().items():
        param.copy_(self.decay * param + (1 - self.decay) * model_state_dict[name])
