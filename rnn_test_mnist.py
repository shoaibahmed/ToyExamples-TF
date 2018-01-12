import numpy as np
import os
import os.path
from optparse import OptionParser

# Command line options
parser = OptionParser()
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-i", "--trainingIterations", action="store", type="int", dest="trainingIterations", default=10000, help="Training iterations")
parser.add_option("-b", "--batchSize", action="store", type="int", dest="batchSize", default=30, help="Batch size")
parser.add_option("-l", "--learningRate", action="store", type="float", dest="learningRate", default=1e-3, help="Learning rate")
parser.add_option("-d", "--dropoutKeepProb", action="store", type="float", dest="dropoutKeepProb", default=0.5, help="Dropout keep probability")
parser.add_option("-n", "--numberOfReadingsPerTimeStamp", action="store", type="int", dest="numberOfReadingsPerTimeStamp", default=28, help="Number of readings per timestamp")
parser.add_option("--numberOfHiddenUnits", action="store", type="int", dest="numberOfHiddenUnits", default=128, help="Number of hidden units")
parser.add_option("--numberOfClasses", action="store", type="int", dest="numberOfClasses", default=10, help="Number of classes")
parser.add_option("--loadSession", action="store_true", dest="loadSession", default=False, help="Load previously saved session")

parser.add_option("--dynamicRNN", action="store_true", dest="dynamicRNN", default=False, help="Use dynamic RNN")
parser.add_option("--biDirectionalLSTM", action="store_true", dest="biDirectionalLSTM", default=False, help="Use Bi-Directional RNN")
parser.add_option("--useRMS", action="store_true", dest="useRMS", default=False, help="Use RMS instead of separate x, y, z values")

# Parse command line options
(options, args) = parser.parse_args()
print (options)

n_sensors = 1
n_input = n_sensors * options.numberOfReadingsPerTimeStamp # Number of sensor readings per timestamp
n_readings_per_timestamp = int(n_input / n_sensors)
n_steps = int(784 / n_readings_per_timestamp) # timesteps (Sequence length)
sequence_length = n_input

print ("Number of inputs: %d" % n_input)
print ("Number of readings per timestamp: %d" % n_readings_per_timestamp)
print ("Number of steps: %d" % n_steps)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

train_data = mnist.train.images
train_labels = mnist.train.labels
validation_data = mnist.validation.images
validation_labels = mnist.validation.labels
test_data = mnist.test.images
test_labels = mnist.test.labels

# Reshape training data
train_data = np.reshape(train_data, (train_data.shape[0], n_steps, n_input))
test_data = np.reshape(test_data, (test_data.shape[0], n_steps, n_input))

if options.trainModel:
    # tf Graph input
    if options.dynamicRNN:
        x = tf.placeholder("float", [None, None, n_input], name="x")
    else:
        x = tf.placeholder("float", [None, n_steps, n_input], name="x")
    
    y = tf.placeholder("float", [None, options.numberOfClasses], name="y")
    dropoutProbilityPlaceholder = tf.placeholder("float", name="dropoutProbilityPlaceholder")

    if options.biDirectionalLSTM:
        num_lstm_outputs = 2 * options.numberOfHiddenUnits
    else:
        num_lstm_outputs = options.numberOfHiddenUnits

    # Define weights
    weights = {
        # Hidden layer weights => 2*options.numberOfHiddenUnits because of forward + backward cells
        # 'out': tf.Variable(tf.random_normal([num_lstm_outputs, options.numberOfClasses]))
        'conv_w': tf.get_variable("conv_w", [n_steps, n_input, 3]),
        'out': tf.get_variable("softmax_w", [num_lstm_outputs, options.numberOfClasses])
    }
    biases = {
        # 'out': tf.Variable(tf.random_normal([options.numberOfClasses]))
        'out': tf.get_variable("softmax_b", [options.numberOfClasses])
    }

    def RNN(x, weights, biases):
    # def RNN(x):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (options.batchSize, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (options.batchSize, n_input)

        # Apply 1-D convolution
        x = tf.nn.conv1d(x, weights['conv_w'], 1, padding='SAME')
        # print (x.get_shape())

        # Unstack to get a list of 'n_steps' tensors of shape (options.batchSize, n_input)
        if not options.dynamicRNN:
            x = tf.unstack(x, n_steps, 1)

            # # X, input shape: (batch_size, time_step_size, input_vec_size)
            # x = tf.transpose(x, [1, 0, 2])  # permute time_step_size and batch_size
            # # XT shape: (time_step_size, batch_size, input_vec_size)
            # x = tf.reshape(x, [-1, n_input]) # each row has input for each lstm cell (lstm_size=input_vec_size)
            # # XR shape: (time_step_size * batch_size, input_vec_size)
            # x = tf.split(x, n_steps, 0) # split them to time_step_size (28 arrays)
            # # Each array shape: (batch_size, input_vec_size)

        if options.biDirectionalLSTM:
            # Define lstm cells with tensorflow
            # Forward direction cell
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(options.numberOfHiddenUnits, forget_bias=1.0)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=dropoutProbilityPlaceholder)
            # Backward direction cell
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(options.numberOfHiddenUnits, forget_bias=1.0)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=dropoutProbilityPlaceholder)

            if options.dynamicRNN:
                outputs, output_states = tf.nn.bidirectional_options.dynamicrnn(lstm_fw_cell, lstm_bw_cell, x, 
                                            sequence_length=[sequence_length] * options.batchSize, dtype=tf.float32)
                outputs = tf.concat(outputs, 2) # Concatenate forward and backward outputs

            else:
                # Get lstm cell output
                try:
                    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                          dtype=tf.float32)
                except Exception: # Old TensorFlow version only returns outputs not states
                    outputs = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                    dtype=tf.float32)
        else:
            # cell = tf.contrib.rnn.BasicLSTMCell(options.numberOfHiddenUnits, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
            # cell = tf.contrib.rnn.NASCell(options.numberOfHiddenUnits, use_biases=True, reuse=tf.get_variable_scope().reuse)
            # cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=dropoutProbilityPlaceholder)
            cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell(options.numberOfHiddenUnits),
                        input_keep_prob=dropoutProbilityPlaceholder) for _ in range(3)], state_is_tuple=False)
            # cell = tf.contrib.rnn.MultiRNNCell([cell] * 5)
            
            outputs, _ = tf.contrib.rnn.static_rnn(cell, x, dtype=tf.float32)
                
        # Linear activation, using rnn inner loop last output
        output = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'], name="logits")
        # output = tf.layers.dense(outputs[-1], options.numberOfClasses, name="logits")
        return output

    pred = RNN(x, weights, biases)
    print (pred.get_shape())
    # pred = RNN(x)

    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=options.learningRate, momentum=0.9).minimize(loss)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    # Tensorboard vis
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)

    # Merge all summaries into a single op
    mergedSummaryOp = tf.summary.merge_all()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    # Initializing the variables
    init = tf.global_variables_initializer()

train_sequences = train_data.shape[0]
test_sequences = test_data.shape[0]

print ("Training sequences: %d" % train_sequences)
print ("Test sequences: %d" % test_sequences)

# Launch the graph
with tf.Session() as sess:
    if options.trainModel:
        os.system("rm -rf ./logs")
        os.system("rm -rf ./model")
        os.system("mkdir ./model")

        sess.run(init)

        # Create summary writer
        summary_writer = tf.summary.FileWriter("./logs", graph=tf.get_default_graph())

        step = 0
        batch_number = 0
        epochs = 0
        sequential_fetch = False

        # Keep training until reach max iterations
        try:
            while step < options.trainingIterations:
                if sequential_fetch:
                    end_index = batch_number + options.batchSize
                    if end_index > train_sequences:
                        end_index = train_sequences
                    indices = np.arange(batch_number, end_index)
                else:
                    indices = np.random.choice(train_sequences, options.batchSize)
                
                data_batch = train_data[indices, :, :]
                # print("Data shape: %s" % str(data_batch.shape))

                # All labels are equivalent
                labels_one_hot = train_labels[indices]

                # Run optimization op (backprop)
                [train_loss, train_acc, summary, _] = sess.run([loss, accuracy, mergedSummaryOp, optimizer], feed_dict={x: data_batch, y: labels_one_hot, dropoutProbilityPlaceholder: options.dropoutKeepProb})
                print ("Iteration: %d, Minibatch Loss: %.3f, Accuracy: %.2f" % (step, train_loss, train_acc * 100))
                
                # Write logs at every iteration
                summary_writer.add_summary(summary, step)

                step += 1

                if sequential_fetch:
                    batch_number = end_index
                    if batch_number >= train_sequences:
                        epochs += 1
                        batch_number = 0

            print("Optimization Finished!")

        except KeyboardInterrupt:
            print("Process interrupted by user")

        # Save final model weights to disk
        saver.save(sess, "model/mnist_classifier")
        print ("Model saved: %s" % ("model/mnist_classifier"))

    if options.testModel:
        saver = tf.train.import_meta_graph("model/mnist_classifier.meta")
        saver.restore(sess, "model/mnist_classifier")

        x = sess.graph.get_tensor_by_name("x:0")
        y = sess.graph.get_tensor_by_name("y:0")
        loss = sess.graph.get_tensor_by_name("loss:0")
        accuracy = sess.graph.get_tensor_by_name("accuracy:0")
        logitsNode = sess.graph.get_tensor_by_name("logits:0")
        dropoutProbilityPlaceholder = sess.graph.get_tensor_by_name("dropoutProbilityPlaceholder:0")

        # Test model's accuracy on test data
        batch_number = 0
        cummulative_accuracy = 0
        step = 0
        options.batchSize = test_sequences # For testing
        num_iter = int(test_sequences / options.batchSize)
        for i in range(num_iter):
            end_index = batch_number + options.batchSize
            if end_index > test_sequences:
                end_index = test_sequences
            indices = np.arange(batch_number, end_index)
            
            data_batch = test_data[indices, :, :]
            # print("Data shape: %s" % str(data_batch.shape))

            # All labels are equivalent
            labels_one_hot = test_labels[indices]
            
            # Compute loss
            [test_loss, test_acc] = sess.run([loss, accuracy], feed_dict={x: data_batch, y: labels_one_hot, dropoutProbilityPlaceholder: 1.0})
            print ("Iteration: %d, Minibatch Loss: %.3f, Accuracy: %.2f" % (step, test_loss, test_acc * 100))
            cummulative_accuracy += test_acc
            
            step += 1
            batch_number = end_index

        cummulative_accuracy = cummulative_accuracy / num_iter
        print ("Test set accuracy: %.2f" % (cummulative_accuracy * 100))