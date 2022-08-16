# Under the Hood!

This repository contains code to run a basic logistic regression model using 1 hidden layer. The number of nodes in the hidden layer can be adjusted using the variable nh defined in the trainNetwork function.

I have only used numpy in this project. To read CSV file for the dataset, i used a standard python library named csv instead of using pandas.

The dataset used in this has been picked up from kaggle and is related to the prediction of heart attack using a set of 15 parameters. Th one dataset contains data of around 4200 cases out of which i have 3800 to train the network and the remaining for testing purposes.

The hyperparameters can be trained further and the accuracy of the model can be increased using those parameters. However, due to lack of time, the maximum accuracy I was able to get on the test set is around **83%**.

## Functions used in this program
All the functions used below are basedn the concept of vectorization. The entire dataset has been been converted to a giant vector, before the performing of operations on it.
#### NormalizeData Function
This function normalizes the data by subtracting the mean and dividing it by the variance. This is done to optimize the gradient descent algorithm so that it can find its minima optimally

#### loadTestData 
Loads the test and training data and makes the required arrays.

#### paramsDefine
Defines the parameters to be used in this application and returns a dictionary of the same.

#### activationFunction
To convert the output produced into required value to be used by the next layer

#### ForwardProps, findCost, backwardProps, updateParameters
Runs the standard algorithm. To improve the problem of reducing the cost function, I am updating the learning rate by dividing it by a factor of 1.1 after every 1500 iterations. this leads to a constantly decreasing cost function improving accuracy and avoid the problem of overshooting the minima in the case of gradient descent.

#### Approach

First the forwardprops is run on the dataset to produce the vectorized output (vector Z1) from the first layer. Then using the activation function it is converted to a vector A1 which is used as input data for the hidden layer. Similarly the same process is applied on the hidden layer to produce the output for the output layer.
Then the output produced by this function is compared with the original output and a final cost of the algorithm is calculated.

Then using this data the derivatives are calculated using the backward props algorithm and the parameters are updated by subtracting the derivatives multiplied by the learning rate. As done in MTH101, in the case of the level function, the direction of gradient gives the direction of increase on function, so the opposite direction gives the direction of decrease of function.

#### trainNetwork, predict
Used to train the network using training data and predict the output using testdata.
