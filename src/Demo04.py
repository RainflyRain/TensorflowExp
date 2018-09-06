from tensorflow.examples.tutorials.mnist import input_data

minist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)

print "Training data size:", minist.train.num_examples

print "Validating data size:", minist.validation.num_examples

print "Testing data size:", minist.test.num_examples

print "Example training data", minist.train.images[0]

print "Example training data label:", minist.train.labels[0]
