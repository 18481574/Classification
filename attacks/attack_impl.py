import torch
import numpy as np 
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from typing import Optional

# attacker from other files
from .pixelAttack import PixelAttacker

__all__ = ['PGD', 'FGSM', 'IFGSM', 'OnePixelAttack', ]

# Random start was not implemented currently

# Modification of the code from https://github.com/Hadisalman/smoothing-adversarial/blob/master/code/attacks.py
class PGD(object):
    """
    PGD attack
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    """

    def __init__(self,
                 steps: int = 10,
                 step_size: float = 0.05,
                 random_start: bool = True,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(PGD, self).__init__()

        self.steps = steps
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            targeted: bool = False) -> torch.Tensor:
        return self._attack(model, inputs, labels, targeted)


    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
             targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        

        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.max_norm/self.steps*2)
        # optimizer = optim.SGD([delta], lr=self.eps_iter)

        for i in range(self.steps):
            adv = inputs + delta
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return inputs + delta

    def set_device(self, device: torch.device):
        self.device = device


class FGSM(object):
    """
    FGSM attack
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    """
    def __init__(self,
                 steps: int = 10,
                 step_size: float = 0.05,
                 random_start: bool = True,
                 device: torch.device = torch.device('cpu')) -> None:
        super(FGSM, self).__init__()

        self.steps = steps
        self.random_start = random_start
        self.device = device


    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            targeted: bool = False) -> torch.Tensor:
        return self._attack(model, inputs, labels, targeted)


    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
             targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.eps_iter)

        for i in range(self.steps):
            adv = inputs + delta
            logits = model(adv)
            pred_labels = logits.argmax(1)
            ce_loss = F.cross_entropy(logits, labels, reduction='sum')
            loss = multiplier * ce_loss

            optimizer.zero_grad()
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

            optimizer.step()

            delta.data.add_(inputs)
            delta.data.clamp_(0, 1).sub_(inputs)

        return inputs + delta
    


    def set_device(self, device: torch.device):
        self.device = device 


class IFGSM(object):
    """
    IFGSM attack
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    device : torch.device, optional
        Device on which to perform the attack.
    """
    def __init__(self,
                 steps: int = 10,
                 step_size: float = 0.05,
                 random_start: bool = True,
                 max_norm: Optional[float] = None,
                 device: torch.device = torch.device('cpu')) -> None:
        super(IFGSM, self).__init__()

        c = step_size * 255.
        self.steps = steps
        self.num_iter = min(round(c+4), round(c*1.25))
        self.random_start = random_start
        self.max_norm = max_norm
        self.device = device


    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            targeted: bool = False) -> torch.Tensor:
        return self._attack(model, inputs, labels, targeted)


    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
             targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            Batch of samples to attack. Values should be in the [0, 1] range.
        labels : torch.Tensor
            Labels of the samples to attack if untargeted, else labels of targets.
        targeted : bool, optional
            Whether to perform a targeted attack or not.
        Returns
        -------
        torch.Tensor
            Batch of samples modified to be adversarial to the model.
        """
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')
        
        batch_size = inputs.shape[0]
        multiplier = 1 if targeted else -1
        delta = torch.zeros_like(inputs, requires_grad=True)

        # Setup optimizers
        optimizer = optim.SGD([delta], lr=self.eps_iter)

        outer_iter = (self.steps + self.num_iter - 1) // self.num_iter 
        inner_iter = self.num_iter
        for i in range(outer_iter):
            for j in range(inner_iter):
                adv = inputs + delta
                logits = model(adv)
                pred_labels = logits.argmax(1)
                ce_loss = F.cross_entropy(logits, labels, reduction='sum')
                loss = multiplier * ce_loss

                optimizer.zero_grad()
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])

                optimizer.step()

                delta.data.add_(inputs)
                delta.data.clamp_(0, 1).sub_(inputs)

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm)

        return inputs + delta
    


    def set_device(self, device: torch.device):
        self.device = device 


def OnePixelAttack(**_info): 
    return PixelAttacker(_info)
    return []

