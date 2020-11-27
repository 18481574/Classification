#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn

# Helper functions
from .differential_evolution import *

__all__ = ['OnePixelAttack']

# Not complete

def perturb_image(xs:np.array , img: torch.Tensor)-> torch.Tensor:
    """
        Get the perturbed imgs with the given pertubation
        Parameters
        ----------
        xs : numpy.array
            Perturbation of each imgm.
        imgs : torch.Tensor
            The original img(only one img this version).
        Returns
        -------
        torch.Tensor
            The polluted imgs with the given perturbation.
        Warnings
        --------
        Pixel value is in [0, 1] in torch.Tensor while [0,255] in numpy format.
        """

    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images

    # guarantee that img size is: dim_x x dim_y x channels -> CWH
    channels_ = img.shape[0]
    N_pixel = 2 + channels_
    tile = [len(xs)] + [1] * 3
    # print(img.shape, tile)
    # imgs = np.tile(img, tile)
    imgs = img.repeat(tile)

    # print(imgs.shape, img.shape, xs.shape)
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // N_pixel)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            # print('Pixel: ', pixel)
            x_pos, y_pos, *color_ = pixel
            # img[x_pos, y_pos] = torch.Tensor(color_ / 255.) 
            for c in range(channels_):
                img[c, x_pos, y_pos] = color_[c]/255.

    return imgs


class OnePixelAttack(object):
    """
    One Pixel attack
    Parameters
    ----------
    steps : int
        Number of steps for the optimization.
    max_norm : float or None, optional
        If specified, the norms of the perturbations will not be greater than this value which might lower success rate.
    popsize: int
        the size of group in DE algorithm
    device : torch.device, optional
        Device on which to perform the attack.
    """
    def __init__(self,
                 steps: int = 100,
                 random_start: bool = True,
                 popsize: int = 200,
                 pixel_count: int = 1,
                 device: torch.device = torch.device('cpu')) -> None:
        super(OnePixelAttack, self).__init__()

        self.steps = steps
        self.random_start = random_start
        self.popsize = popsize
        self.pixel_count = pixel_count
        self.device = device

    def predict_classes(self, xs, img, label_class, targeted, model) -> torch.Tensor:
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = perturb_image(xs, img).to(self.device)
        
        # predictions = model(imgs_perturbed)[:, label_class]
        logit = model(imgs_perturbed)
        predictions = nn.Softmax(dim=1)(logit)[:, label_class]

        # avoid incompatible in codes
        predictions = predictions.detach().cpu().numpy()
        # This function should always be minimized, so return its complement if neede
        return 1 - predictions if targeted else predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = perturb_image(x, img).to(self.device)

        # confidence = model.predict(attack_image)[0]
        confidence = model(attack_image)[0].detach().cpu().numpy()
        predicted_class = np.argmax(confidence).item()

        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
            targeted: bool = False) -> torch.Tensor:
        return self._attack(model, inputs, labels, targeted)

    def _attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor,
             targeted: bool = False) -> torch.Tensor:
        if inputs.min() < 0 or inputs.max() > 1: raise ValueError('Input values should be in the [0, 1] range.')

        Batch_size = inputs.shape[0]
        perturb = torch.zeros_like(inputs)
        
        for k in range(Batch_size):
            # print('{}/{}...'.format(k+1, Batch_size))
            img = inputs[k]
            label_class = labels[k].cpu().item() 

            perturb[k] = self._attack_single(model, img, label_class, targeted)

        return perturb

    def _attack_single(self, model: nn.Module, img: torch.Tensor, label_class: int, targeted: bool) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        Parameters
        ----------
        model : nn.Module
            Model to attack.
        inputs : torch.Tensor
            One img to attack. Values should be in the [0, 1] range.
        label_class : int
            Label of the sample to attack if untargeted, else labels of targets.
            None for untargeted
        Returns
        -------
        torch.Tensor
            An adversarial sample to the model.
        """

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = img.shape[1:3]
        channels_ = img.shape[0]  # inputs: batch x dimx x dimy x channels -> CHW
        pixel_fill = [(0, 256)] * channels_
        bounds = ([(0, dim_x), (0, dim_y)] + pixel_fill) * self.pixel_count 

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, self.popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, img, label_class, targeted, model)

        def callback_fn(x, convergence):
            return self.attack_success(x, img, label_class, model, targeted)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=self.steps, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # # Calculate some useful statistics to return from this function
        # attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        # prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        # predicted_probs = model.predict(np.array([attack_image]))[0]
        # predicted_class = np.argmax(predicted_probs)
        # actual_class = self.y_test[img_id, 0]
        # success = predicted_class != actual_class
        # cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        img_perturbed = perturb_image(attack_result.x, img)[0]

        return img_perturbed
        # return helper.perturb_image(attack_result.x, inputs)
        

    def set_device(device: torch.device):
        self.device = device 


