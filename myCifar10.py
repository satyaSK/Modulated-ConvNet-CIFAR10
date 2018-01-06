import tensorflow as tf
import cifar10_input #This is a helper file I got from the offcial tensorflow website which I've linked in the README(loading data batch-wise)
import cifar10 #This is a helper file I got from the offcial tensorflow website which I've linked in the README(downloading & storing data)
import os
import numpy as np
import time

#Get da Data
cifar10.maybe_download_and_extract()

#Define hyperparameters
batch_size = 128
epochs = 10
learning_rate = 0.001
l,b,channels= 24, 24, 3
n_classes = 10
keep_prob = 0.5
step = 1

with tf.name_scope("DATA"):
	X = tf.placeholder(tf.float32, shape=[batch_size,l,b,channels], name='X')
	Y = tf.placeholder(tf.float32, shape=[batch_size,10], name='Y')

#----------------HELPER CODE(from cifar10_input.py)------------------
def inputs(eval_data=True):
  data_dir = os.path.join('data\cifar10_data', 'cifar-10-batches-bin')
  return cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=batch_size)

def distorted_inputs():
  data_dir = os.path.join('data\cifar10_data', 'cifar-10-batches-bin')
  return cifar10_input.distorted_inputs(data_dir=data_dir, batch_size=batch_size)
#---------------------------------------------

def conv(inputs, filter_shape, bias_shape, padding='SAME'):
	filter_initializer = tf.truncated_normal_initializer()
	bias_initializer = tf.constant_initializer(0)
	filters = tf.get_variable('filter', shape=filter_shape, initializer=filter_initializer)
	biases = tf.get_variable('biases', shape= bias_shape, initializer=bias_initializer)
	convolutions = tf.nn.bias_add(tf.nn.conv2d(inputs, filters, strides=[1,1,1,1], padding=padding), biases)
	normIt = tf.nn.lrn(convolutions, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')# Added normalization
	activated = tf.nn.relu(normIt)
	return activated

def max_pool(feature_maps, ksize=[1,2,2,1] ,strides=[1,2,2,1],padding='SAME'):
	pooled = tf.nn.max_pool(feature_maps, ksize=ksize, strides=strides, padding=padding)
	return pooled

def fully_connected(inputs, weights_shape, biases_shape):
	W_init = tf.truncated_normal_initializer()
	b_init = tf.constant_initializer(0)
	weights = tf.get_variable('weights', shape=weights_shape, initializer=W_init)
	biases = tf.get_variable('biases', shape=biases_shape, initializer=b_init)
	fully_connected = tf.add(tf.matmul(inputs,weights),biases)
	activated_layer = tf.nn.relu(fully_connected)
	return activated_layer

def convNetModel(X, dropout):
	with tf.variable_scope("conv1"):
		conv1 = conv(X, filter_shape=[5,5,3,64], bias_shape=[64])
		pool1 = max_pool(conv1)

	with tf.variable_scope("conv2"):
		conv2 = conv(pool1, filter_shape=[5,5,64,64], bias_shape=[64])
		pool2 = max_pool(conv2)

	with tf.variable_scope("fully_connected_1"):
		output_dimensions = 1
		for d in pool2.get_shape()[1:].as_list():
			output_dimensions *= d

		flattened = tf.reshape(pool2, [-1,output_dimensions])
		fc1 = fully_connected(flattened, weights_shape=[output_dimensions,384], biases_shape= [384])
		fc1_dropout = tf.nn.dropout(fc1,dropout)
	
	with tf.variable_scope("fully_connected_2"):
		fc2 = fully_connected(fc1_dropout, weights_shape=[384, 192], biases_shape=[192])
		fc2_dropout = tf.nn.dropout(fc2, dropout)

	with tf.variable_scope("predictions"):
		prediction_logits = fully_connected(fc2_dropout, weights_shape=[192,10], biases_shape=[10])

	return prediction_logits

def calculate_loss(logits,targets):
	entropy = tf.nn.softmax_cross_entropy_with_logits( logits=logits, labels=targets, name='CE_loss')
	loss = tf.reduce_mean(entropy)
	return loss

def optimize(loss, leanring_rate):
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return optimizer

def get_accuracy(predictions,targets):
	predictions = tf.nn.softmax(predictions)# first get probablity distribution
	correct_predictions = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(targets, axis=1))
	accuracy = tf.reduce_sum(tf.cast(correct_predictions,tf.float32))
	return accuracy

#create model and return logits
logits = convNetModel(X,keep_prob)
#apply softmax and calculate the loss
loss = calculate_loss(logits,Y)
#optimize for an objective function
train_optimizer = optimize(loss, learning_rate)
#calculate accuracy
accuracy_operation = get_accuracy(logits,Y) 


with tf.Session() as sess:
    # Train the Model
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("./Visualize", sess.graph)
	num_batches = int(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/batch_size) 
	X_operation, Y_operation = distorted_inputs()# helper function from cifar10.py
	val_images, val_labels = inputs()#helper function from cifar10.py
	print("\nGood To Go - Training Starts\n")
	for i in range(epochs+1):
	 	epoch_loss = 0
	 	start =time.time()
	 	for _ in range(num_batches):
	 		X_batch, Y_batch = sess.run([X_operation, Y_operation])
	 		e, _,acc = sess.run([loss,train_optimizer,accuracy_operation], feed_dict={X:X_batch,Y:Y_batch})
	 		epoch_loss += e
	 	if i%step==0:
	 		val_x, val_y = sess.run([val_images, val_labels])
	 		val_acc = sess.run(accuracy_operation, feed_dict={X:val_x, Y:val_y})# Do validation
	 		end = time.time()
	 		print("Epoch {0} Training Accuracy: {1:.3f} and ETA = {2:.3f}sec".format(i,(acc/batch_size)*100),(end-start))
	 		print("Validation accuracy now: {0:.3f}\n".format((val_acc/validation_size)*100))
	
	#Testing model
	val_x, val_y = sess.run([val_images, val_labels])
	a = sess.run(accuracy_operation, feed_dict={X:val_x, Y:val_y})# Do validation		
	print("Test accuracy of Model = {0}".format((a/cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)*100))
	writer.close()

### Just for fun ###
	print("Learning rate = {0}\nCross entropy loss fn(Natural choice)\nModel used Adaptive momentum optimizer to minimize loss\n".format(learning_rate.eval()))
	#Tensorboard --logdir = "Visualize"
	Name = "visualize"
	print("To get dataflow graph Use the command-> tensorboard --logdir=\"" + Name + "\" ")