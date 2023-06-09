# Back Propagation 

## Introduction

Back Propagation algorithm also known as "backward propagation of errors," enables the neural networks to adapt and update their weights by propagating the error gradient backward from the output layer to the input layer. Gradients indicate how much each weight contributes to the overall error. Weights are updated backwards to minimize the error.  By doing so, the network can learn from its mistakes and adjust its parameters to improve its performance.

The goal of backpropagation is to minimize the difference between the predicted output of a neural network and the desired output. Starting from the output layer, the error is propagated backward through the network. For each neuron, the algorithm calculates the contribution of that neuron's activations to the overall error. This is done using the chain rule of calculus, which allows us to calculate the derivative of the error with respect to the weights and biases of each neuron.

## Working of Backpropagation

### Forward Propagation: 
The input data is fed through the neural network. Input and output are determined by looking at the problem statement. We don’t know the exact weights initially; we will assign random values to the weights. 

In the diagram above i1 and i2 are the inputs and are connected to h1 using weights w1 and w2. The weighted sum of the inputs is calculated to produce output of each neuron at the hidden layer.

	$h1 = w1*i1 + w2*i2$
	$h2 = w3*i1 + w4*i2$

The output neuron at each hidden layer is introduced with non-linearity. Activation functions are used to add non-linearity to the hidden layer, which helps understand the nonlinear relationships and learn complex patterns. In the formula below a sigmoid function is applied.

	$a_h1 = σ(h1) = 1/ (1 + exp(-h1))$
	$a_h2 = σ(h2) = 1/ (1 + exp(-h2))$

We repeat these calculations for each neuron in each layer, starting from the input layer and moving towards the output layer. In the diagram above, the hidden layer is input to the output layer, hidden layer a_h1 and a_h2 are inputs to output layer o1 using weights w5 and w6 and further activation function is added.
	
	$o1 = w5*a_h1 + w6*a_h2$
	$a_o1 = σ(o1) =1/ (1 + exp(-o1))$
	$o2 = w7*a_h1 + w8*a_h2$
	$a_o2 = σ(o2) =1/ (1+exp(-o2))$
	
## Error Calculation:
The output of the activation function from the output neuron reflects the predicted output of the inputs.  There’s a difference between the predictions and expected output. The error functions tell how close the predicted values are to the expected output. The optimal value for error is zero, meaning there’s no error at all, and both desired and predicted results are identical.

The difference between the predictions and the expected output and calculate the error. The error is quantified using a loss function, such as mean squared error or cross-entropy loss. We calculate the error, then the forward pass ends, and we should start the backward pass to calculate the derivatives and update the parameters. In the diagram above, the loss is calculated by using the formula,
	$E_total = E1 + E2, where E1 is calculated for o1 and E2 is calculated for o2$
	$E1 = ½ * (t1 - a_o1) ²$
	$E2 = ½ * (t2 - a_o2) ²$

