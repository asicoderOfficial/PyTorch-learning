import torch
import logging

logging.basicConfig(level=logging.INFO)

#Create tensor with uninitialized numbers.
x = torch.empty(2,3)

#Create the same tensor, but with fixed datatypes.
#I change it from the default, float32, to float64.
x = torch.empty(2,3, dtype=torch.double)

#Get dimensions of tensor.
logging.info(x.size())

#Create randomly initialized tensors.
x = torch.rand(2,3)
y = torch.rand(2,3)

logging.info(x)
logging.info(y)

#Perform numeric operations.
#They can be called with a function, or using an overloaded operator.
logging.info(x - y)
logging.info(x * y)
logging.info(x / y)

#Slicing. Get all rows except the first one.
#The pattern is as in python: [begin_row, end_row+1 : begin_column, end_column+1]
logging.info(x[1:])

#Reshape. We converted it into a 1d array, by rows.
z = x.view(-1)
logging.info(z)
