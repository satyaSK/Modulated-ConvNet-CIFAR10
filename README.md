# ConvNet for CIFAR-10
This repo contains code to classify RGB CIFAR-10 distorted images. I had already created a CNN model in [this](https://github.com/satyaSK/Tensorflow-Workflow) repository, but it was a beginner level non-reusable code written for basic understanding and usage. The unique way in which the data is structured inside the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, it becomes a surprisingly hard task even for humans to correctly identify the images with high accuracy. More information can be found on the website hosting the dataset.

## Dependencies
* Tensorflow
* Numpy

## Basic Pipeline
```
Input Data(CIFAR-10 RGB distorted Images) -> Convolve(filters) -> Batch Normalization -> Activation(ReLU) -> Max Pooling -> Flatten -> Fully Connected -> Logits(prediction) -> Loss -> Backpropagate to update weights
```

## Why CNN's again?
There were 3 primary reasons as to why I felt the need to pick up this short project/exercise:
* To test the performance of CNN's with a unique dataset of distorted RGB images.
* To structure a CNN in a reusable manner. Here, I have encapsulated all the operations within functions.
* To observe the effectiveness of batch normalization(batch normalization actually works!!).

# What's batch normalization bro?
* batch normalization will apply a transformation, that maintains the mean activation, close to zero, and the activation standard deviation close to one, this allows for faster learning and higher overall accuracy.
* Basically to avoid high fluctuations(w.r.t magnitude) or instability of data flowing through the layers of our neural network, we want to normalize every output feature of every layer, such that it is zero centered.
* Effectively we want the ```mean = 0``` and ```variance = 1```. 
* So on normalizing X(the input features) to take on similar range of values, we can speed up learning. Also, instead of normalizing the inputs only, we want to normalize all the values in our hidden units as well. But wait, there is more to it than this.
* The idea of normalizing logits and normalizing activations is debatable, but a majority of experiments have chosen to normalize logits at a given layer rather than the activations of that layer.
* Batch normalization makes the weights in deeper layers of the network, to be more resistant to the effect of the weights in earlier layers of the network.
* So, if our model learns some ```X -> Y``` mapping, and if the distribution of ```X``` changes, then we might need to retrain the entire model, given we do not normalize the logits at each layer.
* But if we were to normalize our data and the hidden units at each layer, which has an effect of adding noise to our data, then a shift in the distribution of ```X``` might not translate to retraining our model, as our model will have learnt to be more resistant to such noise.
* I've made use of the "local response normalization" in my code which is as easy as calling a single line ```tf.nn.local_response_normalization(<attributes>)``` in tensorflow. However, we can choose to implement a more complex version of batch normalization(which I'm planning to cover a bit later), but for basic learning purposes, we simply use what Tensorflow gives us out of the box.
* The more advanced approach would be that, after normalizing the input ```X``` the result is squashed through a linear function with parameters ```Gamma``` and ```Beta``` which are trainable(I'll cover this in a later update).
* But for now, ```Beta``` and ```Gamma``` are used to scale and shift the normalized distribution. Both of these parameters are learnt.

## Resources
* Checkout these helpful resources I listed below:
	* [Effectiveness of batch normalization](https://medium.com/deeper-learning/glossary-of-deep-learning-batch-normalisation-8266dcd2fa82)
	* [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
	* [What Is Local Response Normalization In Convolutional Neural Networks](https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/)
	
## Basic Usage
For Running, type in terminal
```
python myCifar10.py
```
For the beautiful visualization, type in terminal
```
tensorboard --logdir="visualize"
```



