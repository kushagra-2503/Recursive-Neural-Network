from data import train_data, test_data
import numpy as np
from numpy.random import randn
from RNN import RNN 
import random
#Create the vocabulary.
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size) 

print(enumerate(vocab))

#Assign index to each word
word_to_idx = { w : i for i, w in enumerate(vocab) }
idx_to_word = {i : w for i,w in enumerate(vocab) }

def createInputs(text):
    '''
    Returns an array of one-hot vectors representing the words in the input text    string.
    -text is a string
    -Each one-hot vector has shape (vocab_size,1)
    '''
    inputs = []
    for w in text.split(' '):
        v = np.zeros((vocab_size,1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs
#We will use createInputs later to create vector to pass into our RNN.

def softmax(xs):
    #Applies the softamax function to the input array.
    return np.exp(xs)/sum(np.exp(xs))

#initialize the rnn
rnn = RNN(vocab_size, 2)
inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
probs = softmax(out)

#loop over each training example
for x, y in train_data.items():
    inputs = createInputs(x)
    target = int(y)

#Forward
    out, __ = rnn.forward(inputs)
    probs = softmax(out)

    #Build dL/dy
    d_L_d_y = probs
    d_L_d_y -= 1

    #Backward
    rnn.backprop(d_L_d_y)




def processData(data, backprop = True):
    '''
    Returns the RNN's loss and accuracy for the given data.
   - data is a dictionary mapping text to True or False.
   - backprop determines if the backward phase should be run.
    '''
	items = list(data.items())
	random.shuffle(items)

	loss = 0
	num_correct = 0
	for x,y in items:
		inputs = createInputs(x)
		target = int(y)
	#Forward
	out, __ = rnn.forward(inputs)
	probs = softax(out)
	#Calculate loss/accuracy
	loss = -np.log(probs[target])
	num_correct += int(np.argmax(probs) == target)
	if backprop:
	#build dL/dy
		d_L_d_y = probs
		d_L_d_y[target] -= 1

	#Backprop
	rnn.backprop(d_L_d_y)

	return loss/ len(data), num_correct / len(data)


#training loop
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print('--Epoch %d' %(epoch+1))
        print('Train Loss: %.3f | Accuracy: %.3f' %(train_loss, train_acc))

        test_loss, test_acc = processData(test_data)
        print('Test Loss: %.3f | Accuracy: %.3f' %(test_loss, test_acc))

