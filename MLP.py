# import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import random

#To ensure reproducibility
np.random.seed(11)


# This code defines the inputs and outputs for our four problems, using a task object

class task(object):
    """
    The task object stores the inputs and outputs for a given problem in order to keep things
    consistent between tasks.
    """

    # The initialization function, inputs and outputs must be provided here
    def __init__(self, inputs, outputs):
        """
        Provide a set of inputs and outputs that define the task. Both the inputs and the outputs
        must be a 2D tuple, organized by cases (e.g. data points) on the first dimension, and units
        on the second dimension, e.g. for XOR the first dimension will be 4 in length for both, the
        inputs will have length 2 for the second dimension, and the outputs length 1.
        """

        # store the inputs and outputs as numpy arrays
        self.inputs = inputs
        self.outputs = outputs

        # double check the dimensions are right
        assert self.inputs.shape[0] == self.outputs.shape[0], "Number of cases in input and output not equal."

    # A function to determine the number of cases
    def ncases(self):
        return self.inputs.shape[0]

    # A function to determine the number of input units
    def ninputs(self):
        return self.inputs.shape[1]

    # A function to determine the number of output units
    def noutputs(self):
        return self.outputs.shape[1]


class mlp(object):
    """
    Define a class for a multilayer perceptron here (just one hidden layer).
    You must also define the functions below, and use the arguments provided,
    but you can add additional arguments if you wish.
    Also, note that you are welcome to write your own helper functions.
    Reminder: you should use numpy functions for vector and matrix operations,
    and you have to calculate your own gradients for backprop. No autograd!
    """

    task: task
    nhid: int
    # Matrices for holding the weights of each layer's input (hidden and output layers)
    W_h: np.ndarray
    W_y: np.ndarray

    # The initialization function for the mlp
    def __init__(self, task, nhid, load_weights=False):
        # Keep the task and number of hidden weights available for future use
        self.task = task
        self.nhid = nhid
        if load_weights:
            self.W_h = np.load("W_h.npy")
            self.W_y = np.load("W_y.npy")
        else:
            # Initialize the synaptic weights randomly using a standard normal distribution
            self.W_h = np.random.normal(0, 1, (nhid, task.ninputs()))
            self.W_y = np.random.normal(0, 1, (task.noutputs(), nhid))

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Square error loss function
    def sqr_err(self, e):
        return 0.5 * (np.sum(np.square(e)))

    def feed_forward(self, x, print=False):
        # Input
        new_x: np.ndarray = x
        # In the case of the dimensions being incorrect, add a dimension to input
        if len(new_x.shape) < 2:
            new_x = new_x[np.newaxis].T
        # Output of hidden layer
        o_h: np.ndarray = self.sigmoid(self.W_h.dot(new_x))
        # Output of output layer
        o_y: np.ndarray = self.sigmoid(self.W_y.dot(o_h))
        if print:
            print("Input:")
            print(new_x)
            print("Hidden output:")
            print(o_h)
            print("Output:")
            print(o_y)
        return o_y

    losses: [float]

    def back_prop(self, elem: {'i': int, 'x': np.ndarray}) -> float:
        i: int = elem['i']
        x: np.ndarray = elem['x']
        # In the case of the dimensions being incorrect, add a dimension to input
        if len(x.shape) < 2:
            x = x[np.newaxis].T
        # Output of hidden layer
        o_h: np.ndarray = self.sigmoid(self.W_h.dot(x))
        # Output of output layer
        o_y: np.ndarray = self.sigmoid(self.W_y.dot(o_h))
        # Target output
        t: np.ndarray = self.task.outputs[i]
        # In the case of the dimensions being incorrect, add a dimension to target
        if len(t.shape) < 2:
            t = t[np.newaxis].T
        # Calculate error vector
        e: np.ndarray = t - o_y
        # Pre-calculate a part of the delta_W_y calculation (it will be reused for delta_W_h)
        err_pre_calc: np.ndarray = np.ufunc.reduce(np.multiply, [e, o_y, 1 - o_y])
        # Calculate delta W_y (use matrix multiplication to make it less complicated)
        delta_W_y: np.ndarray = err_pre_calc.dot(o_h.T)
        # Calculate sum of dL/do_y * do_y/dnet_y * dnet_y/do_h
        sum_delta_W_h: np.ndarray = self.W_y.T.dot(err_pre_calc)
        # Calculate delta W_h
        delta_W_h: np.ndarray = np.ufunc.reduce(np.multiply, [sum_delta_W_h, o_h, 1 - o_h]).dot(x.T)
        # Apply weight updates
        self.W_y += delta_W_y
        self.W_h += delta_W_h
        # Add to losses
        self.losses.append(self.sqr_err(e))

    # The function for training the network for one epoch (i.e. one pass through the data)
    def train_one_epoch(self, rate_eta=1) -> float:
        ''' IMPLEMENT LEARNING RATE? '''


        # Randomize order of the training data
        rands = np.random.randint(0,self.task.ninputs(), 1024)
        #print(epoch_inputs)
        train_order: np.ndarray = np.arange(0, len(self.task.inputs))
        np.random.shuffle(train_order)
        # Keep track of all losses for this epoch
        losses: [float] = []
        # Iterate through training data
        for i in rands:
            # Input
            x: np.ndarray = self.task.inputs[i]
            # In the case of the dimensions being incorrect, add a dimension to input
            if len(x.shape) < 2:
                x = x[np.newaxis].T
            # Output of hidden layer
            o_h: np.ndarray = self.sigmoid(self.W_h.dot(x))
            # Output of output layer
            o_y: np.ndarray = self.sigmoid(self.W_y.dot(o_h))
            # Target output
            t: np.ndarray = self.task.outputs[i]
            # In the case of the dimensions being incorrect, add a dimension to target
            if len(t.shape) < 2:
                t = t[np.newaxis].T
            # Calculate error vector
            e: np.ndarray = t - o_y
            # Pre-calculate a part of the delta_W_y calculation (it will be reused for delta_W_h)
            err_pre_calc: np.ndarray = np.ufunc.reduce(np.multiply, [e, o_y, 1 - o_y])
            # Calculate delta W_y (use matrix multiplication to make it less complicated)
            delta_W_y: np.ndarray = err_pre_calc.dot(o_h.T) * rate_eta
            # Calculate sum of dL/do_y * do_y/dnet_y * dnet_y/do_h
            sum_delta_W_h: np.ndarray = self.W_y.T.dot(err_pre_calc)
            # Calculate delta W_h
            delta_W_h: np.ndarray = np.ufunc.reduce(np.multiply, [sum_delta_W_h, o_h, 1 - o_h]).dot(x.T) * rate_eta
            '''if t[1] == 0:
                delta_W_y *= 2
                delta_W_h *= 2
            '''
            # Apply weight updates
            self.W_y += delta_W_y
            self.W_h += delta_W_h
            # Add to losses
            losses.append(self.sqr_err(e))
            '''progress = math.ceil(count / self.task.ncases() * 100)
            if progress % 5 == 0:
                print('\r', f'Training epoch... {progress}%', end='')
            '''

        return np.sum(losses)

    # The function for training the network (returns a list of loss values)
    def train(self, loss_theta = 0.001, rate_eta = 1, n_epochs = 1000000) -> [float]:
        # Initialize tracking variables
        epoch_losses: [float] = []
        start_time: float = time.time()
        last_update_time: float = time.time()
        loss: float = 9999999
        num_epochs: int = 0
        # Train until either number of epochs is reached, or theta is reached
        while loss > loss_theta and num_epochs < n_epochs:
            # Update loss
            loss: float = self.train_one_epoch(rate_eta)
            num_epochs += 1
            epoch_losses.append(loss)
            # Output progress
            if time.time() - last_update_time >= 0.5:
                print('\r', f'Loss: %f - # Epochs: %d' % (loss, num_epochs), end='')
                last_update_time = time.time()
        print('\r', f'Loss: %f - # Epochs: %d - Time: %fs\n' % (loss, num_epochs, time.time() - start_time), end='')
        np.save("./W_y", self.W_y)
        np.save("./W_h", self.W_h)
        return epoch_losses

    # The function for plotting the loss after training
    def plot_loss(self, losses: [float]):
        # Add title and labels
        plt.plot(losses)
        plt.title("Mean Squared Error Loss over Epochs")
        plt.xlabel("Epoch #")
        plt.ylabel("MSE Loss")
        plt.show()

    # The function for plotting the hidden units weights
    def plot_weights(self):
        # Use built-in function to display a matrix
        plt.matshow(self.W_h)
        # Add a color bar to show values
        plt.colorbar()
        # Add title and axes' labels
        plt.title("Hidden Layer Weight Matrix")
        plt.xlabel("Inputs")
        plt.ylabel("Hidden Units")
        plt.show()

    # The function for plotting the hidden unit activity
    def plot_activity(self):
        for input in self.task.inputs:
            A = self.sigmoid(self.W_h.dot(input))
            A = A[np.newaxis].T
            # Use built-in function to display a matrix
            plt.matshow(A)
            # Add a color bar to show values
            plt.colorbar()
            # Add title and axes' labels
            plt.title("Hidden Layer Activation - Input " + str(input))
            plt.xlabel("Activation")
            plt.ylabel("Hidden Units")
            plt.show()

    # The function for plotting the hidden unit activity (aggregate)
    def plot_activity_agg(self):
        # Create initial
        A: np.ndarray = self.sigmoid(self.W_h.dot(self.task.inputs[0]))
        A = A[np.newaxis]
        # Add each activation
        for input in self.task.inputs[1:]:
            A = np.vstack([A, self.sigmoid(self.W_h.dot(input))])
        # Transpose
        A = A.T
        # Use built-in function to display a matrix
        plt.matshow(A)
        # Add a color bar to show values
        plt.colorbar()
        # Add title and axes' labels
        plt.title("Hidden Layer Activation")
        plt.xlabel("Activation")
        plt.ylabel("Hidden Units")
        plt.show()
