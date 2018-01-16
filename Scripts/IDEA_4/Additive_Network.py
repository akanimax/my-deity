"""
    The script that creates a three layer_deep network of the proposed architecture in the
    concept file
"""
from __future__ import print_function
from __future__ import division # this allows for the division to perform true division directly

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

base_data_path = "../../Data/"
mnist_data_path = os.path.join(base_data_path, "MNIST_data")

# define the tensorflow flags mechanism
flags = tf.app.flags
FLAGS = flags.FLAGS

# define the function that consturcts the tensorflow computational graph
def mkGraph(width = 3, depth = 3):
    """ The function that creates and returns the graph required to
    """
    comp_graph = tf.Graph()

    with comp_graph.as_default():
        # step 1: Create the input placeholders for the input to the computation
        tf_input_images = tf.placeholder(tf.float32, shape=(None, None), name="Input_Labels")
        tf_input_labels = tf.placeholder(tf.int32, shape=(None, None), name="Input_Labels")

        # step 2: Create while loop to generate the op for one layer
        def loop_body(count, X):
            # small function for calculating the factorial
            factorial = lambda x: 1 if(x <= 1) else x * factorial(x - 1)

            # create a new variable
            # ??? //TODO Add the matrix multiplication terms to the following computations

            # define the non-linearity
            # return [count + 1, X + (tf.div(tf.pow(X, count), factorial(count)))]

            pass

        tf.while_loop(
            lambda x: tf.less(x, width), # condition

        )

    return comp_graph


def setup_MNIST_data():
    """ Function for setting up the mnist Data
        Uses the tensorflow examples dataset
    """
    print("\nDownloading the dataset (if required) ...")
    mnist_data = input_data.read_data_sets(mnist_data_path, one_hot=True)

    # Create the train_X, train_Y
    train_X = mnist_data.train.images; train_Y = mnist_data.train.labels
    dev_X = mnist_data.validation.images; dev_Y = mnist_data.validation.labels
    test_X = mnist_data.test.images; test_Y = mnist_data.test.labels

    # return these
    return (train_X, train_Y, dev_X, dev_Y, test_X, test_Y)

def main(_):
    """ The main function for binding the app together
    """
    # obtain the mnist data for working with
    train_X, train_Y, dev_X, dev_Y, test_X, test_Y =  setup_MNIST_data()

    # print a description of the obtained data
    print("\n\nObtained Dataset Information")
    print("Training_set shapes:", train_X.shape, train_Y.shape)
    print("Development_set shapes:", dev_X.shape, dev_Y.shape)
    print("Test_set shapes:", test_X.shape, test_Y.shape)

if(__name__ == "__main__"):
    # use the FLAGS mechanism to parse the arguments and test
    flags.DEFINE_integer("network_width", 2,
            "The highest (degree - 1) till which the taylor series is expanded")

    flags.DEFINE_integer("network_depth", 1,
            "The depth of the composition of the polynomials")

    tf.app.run(main)
