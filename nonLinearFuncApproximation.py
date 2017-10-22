import numpy as np

# ML packages
import tensorflow as tf

# Visualization packages
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig, ax = plt.subplots()

# Create the 2-D dataset of points within the range of 0 and 5
numSamples = 200
numDimensions = 2

maxXValue = 10
maxYValue = 7

# maxValue = 10
# trainDataPoints = np.random.random_integers(low=-maxValue, high=maxValue, size=(numSamples, numDimensions))

trainDataPoints = np.array([np.random.random_integers(low=-maxXValue, high=maxXValue, size=(numSamples,)), np.random.random_integers(low=-maxYValue, high=maxYValue, size=(numSamples,))]).T
testDataPoints = np.array([np.random.random_integers(low=-maxXValue, high=maxXValue, size=(numSamples,)), np.random.random_integers(low=-maxYValue, high=maxYValue, size=(numSamples,))]).T

ax.scatter(trainDataPoints[:, 0], trainDataPoints[:, 1], c='g')
# plt.show()

# Desired ellipse params
# (x - x0)^2/a^2 + (y - y0)^2/b^2 = r^2
center = (0, 0)
a = 1
b = 1
r = 5

# boundaryPoints = ((np.square(trainDataPoints[:, 0] - center[0]) / np.square(a)) + (np.square(trainDataPoints[:, 1] - center[1]) / np.square(b)) == np.square(r))
trainLabels = ((np.square(trainDataPoints[:, 0] - center[0]) / np.square(a)) + (np.square(trainDataPoints[:, 1] - center[1]) / np.square(b)) > np.square(r))
testLabels = ((np.square(testDataPoints[:, 0] - center[0]) / np.square(a)) + (np.square(testDataPoints[:, 1] - center[1]) / np.square(b)) > np.square(r))

trainLabelsFloat = trainLabels.astype(float)
testLabelsFloat = trainLabels.astype(float)

outerPoints = trainDataPoints[trainLabels]
# innerPoints = trainDataPoints[[False if currentLabel == True else True for currentLabel in trainLabels]]

# ax.scatter(innerPoints[:, 0], innerPoints[:, 1], c='r')
ax.scatter(outerPoints[:, 0], outerPoints[:, 1], c='r')

ellipse = Ellipse(xy=center, width=(r * 2), height=(r * 2), angle=0.0, fill=False, color='y') # Draw the original ellipse

ax.add_patch(ellipse)
ax.set_title('Train data')
# plt.show()

# Perform polynomial logistic regression
numIterations = 1000
learningRate = 1e-1

dataPointPlaceholder = tf.placeholder(tf.float32, shape=(None, numDimensions), name="dataPointPlaceholder")
labelPlaceholder = tf.placeholder(tf.float32, shape=(None,), name="labelPlaceholder")

# Create the model variables
def addMLPNode(dataInput, outDim):
	W = tf.Variable(tf.random_normal([int(dataInput.get_shape()[-1]), outDim]))
	b = tf.Variable(tf.zeros([outDim]))
	y = tf.matmul(dataInput, W) + b
	return y

# Create the model
hiddenDimension = 10
mlp = tf.nn.tanh(addMLPNode(dataPointPlaceholder, hiddenDimension))
y = tf.squeeze(tf.nn.sigmoid(addMLPNode(mlp, 1)))

# Create the loss
epsilon = 1e-8
with tf.name_scope('loss') as scope:
	loss = - (labelPlaceholder * tf.log(y + epsilon)) - ((1.0 - labelPlaceholder) * tf.log(1.0 - y + epsilon))
	loss = tf.reduce_mean(loss)

with tf.name_scope('optimizer') as scope:
	optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

# Train the model
print ("Training model")
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
	# Initializing the variables
	init = tf.global_variables_initializer()

	sess.run(init)

	for i in range(numIterations):
		_, trainLoss = sess.run([optimizer, loss], feed_dict={dataPointPlaceholder: trainDataPoints, labelPlaceholder: trainLabelsFloat})
		print ("Iteration: %d | Loss: %f" % (i + 1, trainLoss))
	print ("Training finished!")

	# Generate predictions on the test set
	testSetPredictions = sess.run(y, feed_dict={dataPointPlaceholder: testDataPoints})
	testSetPredictions = np.squeeze(testSetPredictions)
	threshold = 0.5
	testSetPredictions = testSetPredictions > threshold

# Create the test set plot
fig2, ax2 = plt.subplots()
ax2.scatter(testDataPoints[:, 0], testDataPoints[:, 1], c='g')

outsidePointsTestSet = testDataPoints[testSetPredictions]
ax2.scatter(outsidePointsTestSet[:, 0], outsidePointsTestSet[:, 1], c='r')

ellipseTwo = Ellipse(xy=center, width=(r * 2), height=(r * 2), angle=0.0, fill=False, color='y') # Draw the original ellipse
ax2.add_patch(ellipseTwo)
ax2.set_title('Test data')

plt.show()