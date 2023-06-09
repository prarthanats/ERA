# Back Propagation 

## Introduction

Back Propagation algorithm also known as "backward propagation of errors," enables the neural networks to adapt and update their weights by propagating the error gradient backward from the output layer to the input layer. Gradients indicate how much each weight contributes to the overall error. Weights are updated backwards to minimize the error.  By doing so, the network can learn from its mistakes and adjust its parameters to improve its performance.

The goal of backpropagation is to minimize the difference between the predicted output of a neural network and the desired output. Starting from the output layer, the error is propagated backward through the network. For each neuron, the algorithm calculates the contribution of that neuron's activations to the overall error. This is done using the chain rule of calculus, which allows us to calculate the derivative of the error with respect to the weights and biases of each neuron.

## Working of Backpropagation
<img width="419" alt="Archietcture" src="https://github.com/prarthanats/ERA/assets/32382676/91b602be-fdd8-4b9d-b481-a3d2bf30e614">

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

## Back Propagation:
For back propagation, we need to understand how the predictions are related to each of the weights so that we calculate the gradients that need to be applied.We calculate the gradients of the error with respect to the weights of each neuron. These gradients indicate how much each weight contributes to the overall error. This tells us the effect of each weight on the prediction error. That is, which parameters do we increase, and which ones do we decrease to get the smallest prediction error. Weights are updated in the opposite direction of the gradients to minimize the error.

In the diagram above, the gradient of total error with respect to weight w5 is calculated using the formula, 

	$∂E_total/∂w5 = ∂ (E1 + E2)/∂w5$
	
Here, since ∂5 has no relation to E2, the derivate of it will be 0. The above equations for backpropagation are derived using the chain rule of calculus.  To know how prediction error changes with respect to weight w5 in the parameters we should find the following intermediate derivatives.
1.Error w.r.t the predicted output (activations of the output layer neurons). This gradient represents how the error changes as we modify the output layer activations (∂E1/∂a_o1)
2.Predicted Output w.r.t the output neuron (output of hidden layer wrt to the weight) (∂a_o1/∂o1)
3.Output neuron wrt to the weight w5. In each hidden layer, we calculate the gradient with respect to the activations and use it to compute the gradients with respect to the weights (∂o1/∂w5)

	$∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5$

In the above diagram,
	
	$E1 = ½ * (t1 - a_o1) ²$, 
	
and the derivative of Error w.r.t the predicted output 
	
	$(a_o1) is ∂E1/∂a_o1 = ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)$

The derivative of Predicted Output(a_o1) w.r.t the output neuron(o1) Is 

	$∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)$, where the derivative of the sigmoid function σ(x) is the sigmoid function σ(x) multiplied by 1−σ(x) *

The derivative of output neuron wrt to the weight w5 is ∂o1/∂w5 = a_h1, Bringing all the derivates together,

	$∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1$

Similar Calculation is done for all the hidden layer weights w6,w7 and w8
	
	$∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2$
	$∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1$
	$∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2$

Similarly, we need to find the gradients for the hidden layer outputs,

	$∂E1/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5$
	$∂E2/∂a_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7$
	$∂E_total/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7$
	$∂E_total/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8$

Similarly, calculations are done on weights of the input layers. 
	
	$∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1$
	$∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2$
	$∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1$
	$∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2$

We have now found the equations using which we can back propagate and adjust the weights.

	$new weight = old weight - (learning rate x weight's gradient)$
	
The forward propagation, backward propagation, and weight updates are repeated for a certain number of iterations (epochs) until the network converges or reaches a desired level of performance.

## Interpreting the results of Backpropagation
1.Gradient Magnitudes: provide insights into how much each weight contributes to the overall error. Larger gradient values indicate that a particular weight or bias has a stronger influence on the error, while smaller gradients suggest a weaker influence.
2.Weight/Derivative Updates: After calculating the gradients, the weights are updated to minimize the error. Positive weight updates indicate an increase in the weight value, while negative updates indicate a decrease. The magnitude of the weight update depends on the learning rate, which determines the step size taken towards minimizing the error.
3.Convergence: monitoring the changes in the error over multiple iterations or epochs, you can assess whether the network is converging. If the error decreases consistently, it suggests that the network is learning and adjusting its weights and biases effectively.
4.Weight Updating: The magnitudes of the updated weights can provide insights into the importance of different features or connections in the network. Larger weight values indicate stronger connections and higher importance, while smaller weights suggest weaker connections.

## Effect of Learning Rate
	
