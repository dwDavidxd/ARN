The code for MNIST.

## How to generate adversarial data for training or testing the networks?
Run "craft_adversarial_examples.py".

We use the "advertorch" toolbox to help generate adversairal samples. This code provide ![](http://latex.codecogs.com/svg.latex?L_{\infty}) PGD, ![](http://latex.codecogs.com/svg.latex?L_{2}) CW, DDN and STA attacks to generate different adversarial samples.

The generated samples can be saved with ".png" and ".npy" format.
