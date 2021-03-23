import numpy as np
from matplotlib import pyplot

def logistic_func(x):
    return 1/(1+np.exp(-x))


"""
1. Try your hand at choosing weights for a network that can compute this function

| Input 1 | Output |
|---------|--------|
| 0       | 1      |
| 1       | 0      |
"""

# remember the second input here is just our bias
input_vector = np.array([0, 1])
input_weights = np.array([6, -3])

activation = logistic_func(np.dot(input_vector, input_weights))
print(activation)









"""
2. Let's make it a bit more complex and add a second input!

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 1      |

 ## Also remember to test your network against all input combinations! ##

E.g.: 
input_vector = np.array([0, 0, 1])
input_vector = np.array([0, 1, 1])
input_vector = np.array([1, 0, 1])
input_vector = np.array([1, 1, 1])
"""


# you'll have to add more inputs (and weights!)
input_vector = np.array([0, 1])
input_weights = np.array([6, -3])

activation = logistic_func(np.dot(input_vector, input_weights))
print(activation)







"""
3. What would you have to do if we wanted to have more than one output?
Let's try to replicate the following truth table (i.e., mirror the bits)

| Input 1 | Input 2 | Output 1 | Output 2 |
|---------|---------|----------|----------|
| 0       | 0       | 0        | 0        |
| 0       | 1       | 1        | 0        |
| 1       | 0       | 0        | 1        |
| 1       | 1       | 1        | 1        |

### NOTE: Because now each input node has more than one output node to connect to, 
we'll need more weights. 

We can store these weights as a matrix with 3 rows (one for each input plus the bias)
and 2 columns (one for each output). Thus, each value is a weight from the row's input 
node to the column's output node. 

"""

input_vector = np.array([1, 0, 1])

#TODO: You'll have to change these weights! 
input_weights = np.array( [[1, 1],
                           [1, 1],
                           [1, 1]])


# https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
# we get to take advantage of more numpy fancyness here,
# when np.dot is given a matrix, it performs matrix multiplication!
activation = logistic_func(np.dot(input_vector, input_weights))
print(np.round(activation, decimals=3))



"""
4. If you're feeling brave, try your hand at this one.
Hint: It's not as simple as it looks... You'll need more **layers** for this one!

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |
"""