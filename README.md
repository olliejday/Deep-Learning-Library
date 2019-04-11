# Deep Learning-Library

A modular Neural Network library. 

### Dependencies:

* Numpy (linear algebra)
* (optional) Matplotlib (for some example plots)


### Usage:

See ```Examples.py``` for example usage.

Keras inspired syntax.

Typical use will consist of:

1. Add an input layer, for ```Ni``` dimensional input vectors:

    ```inputs = Inputs(Ni)```

2. Add some dense (linear) layers with activations:

    ```
    x = Dense(Ni, N1)(inputs)
    x = ReLU()(x)
    x = Dense(N1, N2)(x)
    x = ReLU()(x)
    ```

3. Add an output layer:
 
     ```x = Softmax()(x)```
     
     Or leave it on linear for regression.
 
4. Add a loss.

    Over ```No``` dimension output vectors.
    
    For classification with softmax:
    ```loss = CrossEntropySoftmax(No)```
    
    For regression:
    ```loss = MeanSquaredError(No)```

5. Build the model.
    ```model = Model(inputs=inputs, outputs=x)```
    
6. Add and optimizer to your model and loss.
    ```opt = SGD(model, loss)```
    
7. Train for ```iters``` iterations on ```Xtrn, Ytrn``` dataset.

    ```
    for i in range(iters): 
        opt.step(Xtrn, Ytrn)
    ```
8. Predict on ```Xtst``` test data set.
    ```outputs = model.predict(Xtst)```
    
### Currently Supported

##### Layers
* Inputs
* Dense

##### Weight Initialisers
* Xavier Uniform

##### Bias Initialisers
* Zeroes

##### Activations
* ReLU
* Softmax

##### Losses
* Cross Entropy
* Mean Squared Error

##### Optimizers
* Stochastic gradient descent (SGD)
    
### Improvements

Adding CNN and RNN.
    

 
