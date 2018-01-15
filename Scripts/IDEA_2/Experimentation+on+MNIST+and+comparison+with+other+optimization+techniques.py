
# coding: utf-8

# # In this notebook, I use the update rules according to the last notebook (based on the Newton-Raphson method) on the MNIST dataset.
#
# -------------------------------------------------------------------------------------------------------------------
#
# # This notebook also generates the comparison summary against the prevailing first order (not using hessian) optimization techniques.
#
# -------------------------------------------------------------------------------------------------------------------
#
# ## Technology used: TensorFlow

# start with the usual cells. I don't remember how I attained this habit. Anyway, let's get started:

# In[27]:


# import all the required packages:
# packages used for processing:
from __future__ import print_function # making backward compatible

import matplotlib.pyplot as plt # for visualization
import numpy as np

# THE TensorFlow framework
import tensorflow as tf
# use the tensorflow's archived version of the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow debugger:
from tensorflow.python import debug as tf_debug

# for operating system related stuff
import os
import sys # for memory usage of objects
from subprocess import check_output

# to plot the images inline
# get_ipython().magic(u'matplotlib inline')


# In[28]:


# Input data files are available in the "../Data/" directory.

def exec_command(cmd):
    '''
        function to execute a shell command and see it's
        output in the python console
        @params
        cmd = the command to be executed along with the arguments
              ex: ['ls', '../input']
    '''
    print(check_output(cmd).decode("utf8"))


# In[29]:


# check the structure of the project directory
exec_command(['ls', '../..'])


# In[30]:


# set a seed value for the script
seed_value = 3


# In[31]:


np.random.seed(seed_value) # set this seed for a device independant consistent behaviour


# In[32]:


''' Set the constants for the script '''

# various paths of the files
base_data_path = "../../Data" # the data path

base_model_path = "../../Models/IDEA_2"

# constant values for the script
num_digits = 10 # This is defined. There are 10 labels for 10 digits
img_dim = 28 # images are 28 x 28 sized


# In[33]:


# Hyper parameters for tweaking.
# =================================================================================================================
training_batch_size = 64 # 64 images in each batch
no_of_epochs = 12


# network architecture related parameters:
''' Note that the number of layers will be fixed. you can tweak the number of hidden neurons in these layers: '''
num_hidden_lay_1 = 512
num_hidden_lay_2 = 512
num_hidden_lay_3 = num_digits

# learning rate required for other optimizers:
learning_rate = 3e-4 # lolz! the karpathy constant
# =================================================================================================================


# In[34]:


mnist = input_data.read_data_sets(os.path.join(base_data_path, "MNIST_data"), seed=seed_value, one_hot=True)


# In[35]:


train_X = mnist.train.images; train_Y = mnist.train.labels
dev_X = mnist.validation.images; dev_Y = mnist.validation.labels
test_X = mnist.test.images; test_Y = mnist.test.labels


# In[36]:


# print all the shapes:
print("Training Data shapes: ", train_X.shape, train_Y.shape)
print("Development Data shapes: ", dev_X.shape, dev_Y.shape)
print("Test Data shapes: ", test_X.shape, test_Y.shape)


# In[37]:


# define the total_train_examples, total_dev_examples and the total_test_examples using the above arrays
total_train_examples = train_X.shape[0]
total_dev_examples = dev_X.shape[0]
total_test_examples = test_X.shape[0]
input_dimension = train_X.shape[1]


# In[38]:


# just double checking if all the values are correct:
print("Training_data_size =", total_train_examples)
print("Development_data_size =", total_dev_examples)
print("Test_data_size =", total_test_examples)
print("Input data dimensions =", input_dimension)


# the following is a randomized cell for visualizing the input data. This is just to verify the sanity of the data

# In[39]:


''' Randomized cell: Behaviour changes upon running multiple times '''

random_index = np.random.randint(total_train_examples)

# bring the random image from the training data
random_image = train_X[random_index].reshape((img_dim, img_dim))
label_for_random_image = np.argmax(train_Y[random_index])

# display this random image:
plt.figure().suptitle("Image for digit: " + str(label_for_random_image))
plt.imshow(random_image);


# ### seems like we have randomly landed on a mislabelled example! That's the magic of the random_seed_value = 3
#
# don't worry! There is no jumbling in the labels. Run the above cell more times and you will see that it was just that one example that is mis labelled in the original dataset itself.

# # Experimentation:

# Create the neural network graph for classifying the mnist dataset. This is going to be a very simple graph. With just three fully connected layers and there is no regularization (L2 or dropouts).

# point to restart the graph building process:
#

# In[40]:


tf.reset_default_graph()


# In[41]:


layer1 = tf.layers.Dense(
            units = num_hidden_lay_1,
            activation = tf.nn.selu,
            use_bias = True,
            kernel_initializer = tf.contrib.layers.xavier_initializer(seed = seed_value),
            bias_initializer = tf.contrib.layers.xavier_initializer(seed = seed_value),
            #kernel_constraint = tf.keras.constraints.MaxNorm(max_value=1), # unit norm constraint
            name = "fully_connected"
         )

layer2 = tf.layers.Dense(
            units = num_hidden_lay_2,
            activation = tf.nn.selu,
            use_bias = True,
            kernel_initializer = tf.contrib.layers.xavier_initializer(seed = seed_value),
            bias_initializer = tf.contrib.layers.xavier_initializer(seed = seed_value),
            #kernel_constraint = tf.keras.constraints.MaxNorm(max_value=1), # unit norm constraint
            name = "fully_connected"
         )

layer3 = tf.layers.Dense(
            units = num_hidden_lay_3,
            activation = None, # note that here we apply the softmax nonlinearity for obtaining the probabilities
            use_bias = True,
            kernel_initializer = tf.contrib.layers.xavier_initializer(seed = seed_value),
            bias_initializer = tf.contrib.layers.xavier_initializer(seed = seed_value),
            #kernel_constraint = tf.keras.constraints.MaxNorm(max_value=1), # unit norm constraint
            name = "fully_connected"
         )


# define the input placeholders for the computations

# In[42]:


with tf.name_scope("Input_Placeholders"):
    tf_input_images = tf.placeholder(tf.float32, shape=(None, input_dimension), name="input_images")
    tf_input_labels = tf.placeholder(tf.float32, shape=(None, num_digits), name="input_labels")


# define the neural_network computations layer by layer:

# In[43]:


with tf.variable_scope("Layer_1"):
    lay_1_out = layer1(tf_input_images)


# In[44]:


with tf.variable_scope("Layer_2"):
    lay_2_out = layer2(lay_1_out)


# In[45]:


with tf.variable_scope("Layer_3"):
    lay_3_out = layer3(lay_2_out)


# check the dimensionality of the lay_3_output

# In[46]:


lay_3_out


# The shape is as expected:

# In[47]:


# extract all the weights and biases from the layers for summary generation
lay_1_kernel, lay_1_biases = layer1.weights
lay_2_kernel, lay_2_biases = layer2.weights
lay_3_kernel, lay_3_biases = layer3.weights

# attach summary op to all the trainable weights and biases of the three layers:
lay_1_k_sum = tf.summary.histogram("l1_k", lay_1_kernel); lay_1_b_sum = tf.summary.histogram("l1_b", lay_1_biases)
lay_2_k_sum = tf.summary.histogram("l2_k", lay_2_kernel); lay_2_b_sum = tf.summary.histogram("l2_b", lay_2_biases)
lay_3_k_sum = tf.summary.histogram("l3_k", lay_3_kernel); lay_3_b_sum = tf.summary.histogram("l3_b", lay_3_biases)


# In[48]:


# define the predictions:
with tf.name_scope("Predictions"):
    predictions = tf.nn.softmax(lay_3_out)


# In[49]:


# define the loss for the model
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_input_labels, logits=lay_3_out))

    # add a scalar summary for the loss:
    loss_summary = tf.summary.scalar("Loss", loss)


# In[50]:


# define the errands for this model:
with tf.name_scope("Errands"):
    all_summaries = tf.summary.merge_all()


# ## session and runners:

# In[25]:


# define a pseudo session to generate the visualizer and for printing all the trainable variables in the model
# with tf.Session() as sess:
#     # create the tensorboard writer:
#     tensorboard_writer = tf.summary.FileWriter(os.path.join(base_model_path, "Visualizer"), graph=sess.graph,
#                                               filename_suffix = ".bot")
#
#     # init the session:
#     sess.run(tf.global_variables_initializer())
#
#     # print all the trainable variables in the graph:
#     tvars = tf.trainable_variables()
#     tvars_vals = sess.run(tvars)
#
#     for var, val in zip(tvars, tvars_vals):
#         print(var.name)


# Run tensorboard with the Models/IDEA_2 as the logdir to view all these visualizations:

# In[54]:


# define the function for training the above model along with the optimizer as an argument to it:
def train(X, Y, batch_size, no_of_epochs, optimizing_step, model_path, model_name, debug=False):
    '''
        Function to train the above generated graph using the optimizing step privided

        Arguments:
        X, Y = The data to train on.
        batch_size = size of each minibatch
        no_of_epochs = no of epochs to train for
        optimizing_step = the tensorflow op that optimizes the weights
        model_path = The path where the trained model is to be saved
        model_name = The name of the model for saving
        debug = boolean controlling if it needs to be a debugging session
    '''

    # This is a temporary code and not the full blown:
    sess = tf.InteractiveSession()

    if(debug):
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    # create a tensorboard writer
    tensorboard_writer = tf.summary.FileWriter(logdir=model_path,
                                               graph=sess.graph, filename_suffix=".bot")

    # create a saver
    saver = tf.train.Saver(max_to_keep=3)

    # restore the session if the checkpoint exists:
    if(os.path.isfile(os.path.join(model_path, "checkpoint"))):
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

    else: # initialize all the variables:
        sess.run(tf.global_variables_initializer())

    global_step = 0
    print("Starting the training process . . .")
    for epoch in range(no_of_epochs):

        # run through the batches of the data:
        for batch in range(int(np.ceil(float(total_train_examples) / batch_size))):
            start = batch * batch_size; end = start + batch_size

            # extract the relevant data:
            batch_data_X = X[start: end]
            batch_data_Y = Y[start: end]

            _, cost, sums = sess.run([optimizing_step, loss, all_summaries],
                                                    feed_dict={tf_input_images: batch_data_X,
                                                              tf_input_labels: batch_data_Y})

            # save the summaries
            if(batch % 50 == 0):
                tensorboard_writer.add_summary(sums, global_step)

            # increment the global step
            global_step += 1

        print("epoch = ", epoch, "cost = ", cost)

        # save the model after every epoch
        saver.save(sess, os.path.join(model_path, model_name), global_step=(epoch + 1))

    # Once, the training is complete:
    print("Training complete . . .")

    # close the session ->
    sess.close()


# # Train using the Adam optimizer:

# In[27]:


# define the Adam trainer for the model defined
with tf.name_scope("Adam_Trainer"):
    train_step_adam = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# In[28]:


model_name = "Adam_Model_with_SeLU"
model_save_path = os.path.join(base_model_path, model_name)


# In[29]:


train(train_X, train_Y, training_batch_size, no_of_epochs, train_step_adam, model_save_path, model_name)


# # Train using the Adadelta Optimizer

# In[31]:


# # define the Adadelta trainer for the model defined
# with tf.name_scope("Adadelta_Trainer"):
#     train_step_adadelta = tf.train.AdadeltaOptimizer().minimize(loss) # using the default learning rate
#
# model_name = "Adadelta_Model"
# model_save_path = os.path.join(base_model_path, model_name)
#
# train(train_X, train_Y, training_batch_size, no_of_epochs, train_step_adadelta, model_save_path, model_name)
#
#
# # # Train using the Adagrad Optimizer
#
# # In[33]:
#
#
# # define the Adagrad trainer for the model defined
# with tf.name_scope("Adagrad_Trainer"):
#     train_step_adagrad = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
#
# model_name = "Adagrad_Model"
# model_save_path = os.path.join(base_model_path, model_name)
#
# train(train_X, train_Y, training_batch_size, no_of_epochs, train_step_adagrad, model_save_path, model_name)
#
#
# # # Train using the GradientDescentOptimizer
#
# # In[34]:
#
#
# # define the GradientDescent trainer for the model defined
# with tf.name_scope("GradientDescent_Trainer"):
#     train_step_gradDesc = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#
# model_name = "GradientDescent_Model"
# model_save_path = os.path.join(base_model_path, model_name)
#
# train(train_X, train_Y, training_batch_size, no_of_epochs, train_step_gradDesc, model_save_path, model_name)
#
#
# # # Train using MomentumOptimizer
#
# # In[35]:
#
#
# # define the Momentum trainer for the model defined
# with tf.name_scope("Momentum_Trainer"):
#     train_step_momentum = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(loss)
#
# model_name = "MomentumOptimizer_Model"
# model_save_path = os.path.join(base_model_path, model_name)
#
# train(train_X, train_Y, training_batch_size, no_of_epochs, train_step_momentum, model_save_path, model_name)
#
#
# # # Train using RMSPropOptimizer
#
# # In[36]:
#
#
# # define the Momentum trainer for the model defined
# with tf.name_scope("RMSProp_Trainer"):
#     train_step_rmsprop = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
#
# model_name = "RMSpropOptimizer_Model"
# model_save_path = os.path.join(base_model_path, model_name)
#
# train(train_X, train_Y, training_batch_size, no_of_epochs, train_step_rmsprop, model_save_path, model_name)
#
#
# # # Train using the RANIK_Optimizer
#
# # In[52]:


# # The op used for defining the optimizer:
# with tf.name_scope("ranik"):
#     # obtain the gradients of the cost wrt all the three layers' parameters
#     with tf.name_scope("gradients"):
#         gra_lay_1_kernel, gra_lay_1_biases = tf.gradients(loss, [lay_1_kernel, lay_1_biases])
#         gra_lay_2_kernel, gra_lay_2_biases = tf.gradients(loss, [lay_2_kernel, lay_2_biases])
#         gra_lay_3_kernel, gra_lay_3_biases = tf.gradients(loss, [lay_3_kernel, lay_3_biases])
#
#     # define the ops for updating the three layers kernels and biases
#     with tf.name_scope("update"):
#         op1=tf.assign(lay_1_kernel, lay_1_kernel-((loss * gra_lay_1_kernel) / (loss + tf.square(gra_lay_1_kernel))))
#         op2=tf.assign(lay_1_biases, lay_1_biases-((loss * gra_lay_1_biases) / (loss + tf.square(gra_lay_1_biases))))
#         op3=tf.assign(lay_2_kernel, lay_2_kernel-((loss * gra_lay_2_kernel) / (loss + tf.square(gra_lay_2_kernel))))
#         op4=tf.assign(lay_2_biases, lay_2_biases-((loss * gra_lay_2_biases) / (loss + tf.square(gra_lay_2_biases))))
#         op5=tf.assign(lay_3_kernel, lay_3_kernel-((loss * gra_lay_3_kernel) / (loss + tf.square(gra_lay_3_kernel))))
#         op6=tf.assign(lay_3_biases, lay_3_biases-((loss * gra_lay_3_biases) / (loss + tf.square(gra_lay_3_biases))))
#
#         # group all the 6 ops into one
#         update_step = tf.group(op1, op2, op3, op4, op5, op6, name="combined_update")
#
#
# # In[55]:
#
#
# # define the Ranik trainer for the model defined
# with tf.name_scope("Ranik_Trainer"):
#     train_step_ranik = update_step
#
# model_name = "RanikOptimizer_xavier_initialization_and_selu_activation_Model"
# model_save_path = os.path.join(base_model_path, model_name)
#
# train(train_X, train_Y, training_batch_size, no_of_epochs, train_step_ranik, model_save_path, model_name, debug = True)
