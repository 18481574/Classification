#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
import pickle

import torch

# Helper functions
from differential_evolution import *

__all__ = ['PixelAttacker']

# Not complete
def perturb_image(xs:numpy.array , img: torch.Tensor)-> torch.Tensor:
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    # print(img.shape, tile)
    imgs = np.tile(img, tile)
    # print(imgs.shape, img.shape, xs.shape)
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 3)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            # print('Pixel: ', pixel)
            x_pos, y_pos, *rgb = pixel
            # print(x_pos, y_pos, rgb)
            # print(type(rgb), rgb)
            img[x_pos, y_pos] = rgb 
            img[x_pos, y_pos] /= 255.

    return imgs


class PixelAttacker(object):
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
                 steps: int = 10,
                 random_start: bool = True,
                 popsize: int = 400,
                 device: torch.device = torch.device('cpu')) -> None:
        super(PixelAttacker, self).__init__()

        self.steps = steps
        self.random_start = random_start
        self.popsize = popsize

        self.device = device


    def predict_classes(self, xs, imgs, labels, targeted, model):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = perturb_image(xs, imgs)
        mask = _get_mask(labels, targeted)
        predictions = model(imgs_perturbed)
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

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

        # Change the target class based on whether this is a targeted attack or not
        # targeted_attack = target is not None
        # target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = inputs.shape[1:3]
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count * inputs.shape[0]

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, labels, targeted, model)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, model, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]


        return helper.perturb_image(attack_result.x, inputs)
        

    def set_device(device: torch.device):
        self.device = device 




class PixelAttacker:
    def __init__(self, models, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(self.models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, img_id, model, target=None, pixel_count=1,
               maxiter=75, popsize=400, verbose=False, plot=False):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, model, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, model, targeted_attack, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(
            predict_fn, bounds, maxiter=maxiter, popsize=popmul,
            recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x]

    def attack_all(self, models, samples=500, pixels=(1, 3, 5), targeted=False,
                   maxiter=75, popsize=400, verbose=False):
        results = []
        for model in models:
            model_results = []
            valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img
            img_samples = np.random.choice(valid_imgs, samples)

            for pixel_count in pixels:
                for i, img in enumerate(img_samples):
                    print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                    targets = [None] if not targeted else range(10)

                    for target in targets:
                        if targeted:
                            print('Attacking with target', self.class_names[target])
                            if target == self.y_test[img, 0]:
                                continue
                        result = self.attack(img, model, target, pixel_count,
                                             maxiter=maxiter, popsize=popsize,
                                             verbose=verbose)
                        model_results.append(result)

            results += model_results
            helper.checkpoint(results, targeted)
        return results

