import torch

def softmax_temp(x, temp):
    dividend = torch.exp(x/ temp)
    divisor = torch.sum(torch.exp(x/temp))
    return dividend/ divisor

def pad_sequence(observations, padding_side="left"):
  observations = [obs.squeeze(0) for obs in observations]
  return torch.nn.utils.rnn.pad_sequence([
       torch.flip(obs, [0]) for obs in observations
    ]).flip(0).T