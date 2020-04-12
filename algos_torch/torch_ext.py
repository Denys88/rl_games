import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def save_scheckpoint(filename, epoch, model, optimizer):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
    torch.save(state, filename + '.pth')

def load_checkpoint(filename, model, optimizer):
    start_epoch = 0
    print("=> loading checkpoint '{}'".format(filename + '.pth'))
    checkpoint = torch.load(filename)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    return epoch
