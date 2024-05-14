# PINNHeat: Physics-Informed Neural Networks for Heat Equation Modeling

PINNHeat is a project that demonstrates the use of Physics-Informed Neural Networks (PINNs) to solve the heat equation. This project utilizes PyTorch to implement a neural network model that learns the solution to the heat equation and generates predictions for temperature distribution over a given domain.

## Training Progress

![heat_equation_training](https://github.com/Sushant369/Physics-Informed-Neural-Networks-for-Heat-Equation/assets/72655705/5c126312-d1a9-4ad1-91a6-c6c460096d01)
The above GIF shows the training progress of the PINN model, demonstrating how the predicted temperature distribution evolves over epochs.

![Figure_1](https://github.com/Sushant369/Physics-Informed-Neural-Networks-for-Heat-Equation/assets/72655705/aef046e4-f86d-47ac-ad7b-955730458333)

## Overview

The heat equation is a fundamental partial differential equation that describes the distribution of heat (or temperature) over time in a given region. It is widely applicable in various fields, including physics, engineering, and materials science. Solving the heat equation analytically can be challenging for complex geometries and boundary conditions. Therefore, numerical methods, such as finite difference or finite element methods, are often used for practical solutions.

PINNs offer an alternative approach by combining neural networks with the physics of the problem, thereby leveraging data-driven learning to approximate the solution. This project implements a PINN model using a neural network architecture to learn the solution to the heat equation, incorporating initial and boundary conditions as well as physics-based constraints.

## Features

- Implementation of a PINN model using PyTorch.
- Training the model to learn the solution to the heat equation with given initial and boundary conditions.
- Visualization of the learned temperature distribution over the domain.
- Generation of an animated GIF showing the training progress.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Imageio

## Usage

1. Install the required dependencies using pip:

```bash
pip install torch numpy matplotlib imageio


