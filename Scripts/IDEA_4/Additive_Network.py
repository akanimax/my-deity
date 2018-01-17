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
base_model_path = "../../Models/IDEA_4/"

# define the tensorflow flags mechanism
flags = tf.app.flags
FLAGS = flags.FLAGS

# define the function that consturcts the tensorflow computational graph
def mk_graph(img_dim, num_labels, poly_width = 3, depth = 3, hidd_repr_size = 512):
    """ The function that creates and returns the graph required to
        img_dim = image dimensions (Note, that the image needs to be flattened out before feeding here)
        num_labels = no_of classes to classify into
    """
    comp_graph = tf.Graph()

    with comp_graph.as_default():
        # step 1: Create the input placeholders for the input to the computation
        with tf.name_scope("Input"):
            tf_input_images = tf.placeholder(tf.float32, shape=(None, img_dim), name="Input_Labels")
            tf_input_labels = tf.placeholder(tf.float32, shape=(None, num_labels), name="Input_Labels")

        print("\nInput Placeholder Tensors:", tf_input_images, tf_input_labels)

        # step 2: Construct the network architecture based on the width and the depth specified
        # Note that this is static graph creation
        # There doesn't seem to be any reason for dynamic graph building
        def neural_layer(input, out_dim, step):
            """ The method that defines a single neural layer
            """
            # method to calculate the factorial of a number
            factorial = lambda x: 1 if(x <= 1) else x * factorial(x - 1)

            with tf.variable_scope("neural_layer"+str(step)):
                # create the variable tensors ->
                # additive bias
                bias = tf.get_variable("bias", shape=(out_dim), initializer=tf.zeros_initializer())

                # additive weight transformations
                inp_dim = input.get_shape()[-1]
                weights = [tf.get_variable("weight"+str(i), shape=(inp_dim, out_dim),
                            initializer=tf.contrib.layers.xavier_initializer(seed = FLAGS.seed_value))
                            for i in range(1, poly_width)]

                # attach the summary ops to the biases and weights
                bias_summary = tf.summary.histogram("Layer"+str(step)+"/bias", bias)
                weights_summary = [tf.summary.histogram("Layer"+str(step)+"/"+weight.name, weight)
                            for weight in weights]

                # define the compuataion ops for this layer
                out = bias # initialize the output tensor
                for degree in range(1, poly_width):
                    out = out + tf.matmul(tf.pow(input, degree) / factorial(degree), weights[degree - 1])

            return out # return the calculated tensor

        if(depth > 1):
            lay1_out = neural_layer(tf_input_images, hidd_repr_size, 1)
        else:
            lay1_out = neural_layer(tf_input_images, num_labels, 1)

        # define the while loop for creating the hidden layer computations
        lay_out = lay1_out # initialize to output of first layer
        for lay_no in range(2, depth):
            lay_out = neural_layer(lay_out, hidd_repr_size, lay_no)

        # define the output layer
        if(depth > 1):
            output = neural_layer(lay_out, num_labels, depth)
        else:
            output = lay1_out

        print("Final output:", output)

    return comp_graph, {"output": output, "labels": tf_input_labels, "input": tf_input_images}


def setup_MNIST_data():
    """ Function for setting up the mnist Data
        Uses the tensorflow examples dataset
    """
    print("\nDownloading the dataset (if required) ...")
    mnist_data = input_data.read_data_sets(mnist_data_path, one_hot=True, )

    # Create the train_X, train_Y
    train_X = mnist_data.train.images; train_Y = mnist_data.train.labels
    dev_X = mnist_data.validation.images; dev_Y = mnist_data.validation.labels
    test_X = mnist_data.test.images; test_Y = mnist_data.test.labels

    # return these
    return (train_X, train_Y, dev_X, dev_Y, test_X, test_Y)

def main(_):
    """ The main function for binding the app together
    """

    """
    ============================================================================
    || HYPERPARAMETERS TWEAKABLE THROUGH COMMAND_LINE ARGS
    ============================================================================
    """
    model_name = str(FLAGS.network_depth) + "-deep-"+str(FLAGS.network_width) + "-wide-"
    model_name += str(FLAGS.hidden_representation_size) + "-hdr-" + str(FLAGS.epochs) + "-epochs"

    no_of_epochs = FLAGS.epochs
    learning_rate = FLAGS.learning_rate
    training_batch_size = FLAGS.batch_size
    """
    ============================================================================
    """

    # obtain the mnist data for working with
    train_X, train_Y, dev_X, dev_Y, test_X, test_Y =  setup_MNIST_data()

    total_train_examples = train_X.shape[0]

    # print a description of the obtained data
    print("\n\nObtained Dataset Information")
    print("Training_set shapes:", train_X.shape, train_Y.shape)
    print("Development_set shapes:", dev_X.shape, dev_Y.shape)
    print("Test_set shapes:", test_X.shape, test_Y.shape)

    # get the computation Graph
    cmp_graph, int_dict = mk_graph(train_X.shape[-1], train_Y.shape[-1],
                        FLAGS.network_width, FLAGS.network_depth, FLAGS.hidden_representation_size)

    # add the training and runner ops to this computation graph
    with cmp_graph.as_default():
        # define the predictions from the output tensor
        output = int_dict["output"]; labels = int_dict["labels"]
        tf_input_images = int_dict["input"]

        with tf.name_scope("Predictions"):
            predictions = tf.nn.softmax(output) # obtain the softmax

        with tf.name_scope("Loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=labels))

            loss_summary = tf.summary.scalar("Loss", loss)

        # define the trainer
        with tf.name_scope("Trainer"):
            optmizer = tf.train.AdamOptimizer(learning_rate)
            train_step = optmizer.minimize(loss)

        # define the accuracy
        with tf.name_scope("Accuracy"):
            correct = tf.equal(tf.argmax(predictions, axis=-1), tf.argmax(labels, axis=-1))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.cast(tf.shape(labels)[0], tf.float32)

            accuracy_summary = tf.summary.scalar("Accuracy", accuracy)

        # finally define the required errands:
        with tf.name_scope("Errands"):
            init = tf.global_variables_initializer()
            all_sums = tf.summary.merge_all()

    # Run the Session for training the graph
    with tf.Session(graph=cmp_graph) as sess:
        # create a tensorboard writer
        model_save_path = os.path.join(base_model_path, model_name)
        tensorboard_writer = tf.summary.FileWriter(logdir=model_save_path, graph=sess.graph, filename_suffix=".bot")

        # create a saver
        saver = tf.train.Saver(max_to_keep=2)

        # restore the session if the checkpoint exists:
        if(os.path.isfile(os.path.join(model_save_path, "checkpoint"))):
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

        else: # initialize all the variables:
            sess.run(init)

        global_step = 0
        print("Starting the training process . . .")
        for epoch in range(no_of_epochs):
            # run through the batches of the data:
            accuracies = [] # initialize this to an empty list
            runs = int((total_train_examples / training_batch_size) + 0.5)
            checkpoint = runs / 10
            for batch in range(runs):
                start = batch * training_batch_size; end = start + training_batch_size

                # extract the relevant data:
                batch_data_X = train_X[start: end]
                batch_data_Y = train_Y[start: end]

                # This is batch gradient descent: (We are running it only on first 512 images)
                _, cost, acc, sums = sess.run([train_step, loss, accuracy, all_sums],
                                                        feed_dict={tf_input_images: batch_data_X,
                                                                  labels: batch_data_Y})

                # append the acc to the accuracies list
                accuracies.append(acc)

                # save the summarys
                if(batch % checkpoint == 0):
                    tensorboard_writer.add_summary(sums, global_step)

                # increment the global step
                global_step += 1

            print("\nepoch = ", epoch, "cost = ", cost)

            # evaluate the accuracy of the whole dataset:
            print("accuracy = ", sum(accuracies) / len(accuracies))
            # evaluate the accuracy for the dev set
            dev_acc = sess.run(accuracy, feed_dict={tf_input_images: dev_X, labels: dev_Y})
            print("dev_accuracy = ", dev_acc)

            # save the model after every epoch
            saver.save(sess, os.path.join(model_save_path, model_name), global_step=(epoch + 10))

        # Once, the training is complete:
        # print the test accuracy:
        acc = sess.run(accuracy, feed_dict={tf_input_images: test_X, labels: test_Y})
        print("Training complete . . .")
        print("Obtained Test accuracy = ", acc)

if(__name__ == "__main__"):
    # use the FLAGS mechanism to parse the arguments and test
    flags.DEFINE_integer("network_width", 2,
            "The highest (degree - 1) till which the taylor series is expanded")

    flags.DEFINE_integer("network_depth", 1,
            "The depth of the composition of the polynomials")

    flags.DEFINE_integer("hidden_representation_size", 512,
            "The size (dimensionality) of the hidden representations")

    flags.DEFINE_integer("seed_value", 3,
            "The seed value for initialization of the weight matrices")

    flags.DEFINE_integer("epochs", 12,
            "The number of epochs for which the model is to be trained")

    flags.DEFINE_integer("learning_rate", 3e-4,
            "The learning rate for Adam Optimizer")

    flags.DEFINE_integer("batch_size", 64,
            "The batch size for Adam in SGD settings")

    tf.app.run(main)
