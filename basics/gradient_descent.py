import torch
import logging

logging.basicConfig(level=logging.INFO)

#NOTE
#The parameter requires_grad, is False by default (except if it is wrapped in a nn.Parameter).
x = torch.randn(2, requires_grad=True)
logging.info(x)

#Create operation which will be performed as a node, to let PyTorch create the computational graph.
#That way, a single value for each node is assigned, and backpropagation can be applied.
y = (x ** 2)
v = torch.randn(2)
logging.info(y)

y.backward(v) #J * v -> multiplication of the jacobian matrix by the gradient vector.

logging.info(x.grad)

#Let's create a simple model!!

#Weights matrix.
w = torch.rand(8, requires_grad=True)
#Feature vector.
x = torch.rand(8)

for epoch in range(10):
    #The function computed is the multiplication.
    #Forward propagation, get results with weights
    forward_prop_result = (w * x).mean()
    #Backward propagation, optimize.
    forward_prop_result.backward()
    logging.info(f'Epoch {epoch}: {w.grad}') 
    #Empty gradient.
    w.grad.zero_()