The code for MNIST.

## How to generate adversarial data for training or testing the networks?
Run "craft_adversarial_examples.py".

We use the "advertorch"(https://github.com/BorealisAI/advertorch) toolbox to help generate adversairal samples. This code provides ![](http://latex.codecogs.com/svg.latex?L_{\infty}) PGD, ![](http://latex.codecogs.com/svg.latex?L_{2}) CW, DDN(https://arxiv.org/abs/1811.09600) and STA(https://openreview.net/forum?id=HyydRMZC-) attacks to generate different adversarial samples.

The generated samples can be saved with ".png" and ".npy" format.
