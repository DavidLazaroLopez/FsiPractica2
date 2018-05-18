# -*- coding: utf-8 -*-
import gzip
import cPickle

import matplotlib.pyplot as plt



import tensorflow as tf
import numpy as np

import time


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h



f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y   = test_set


train_y = one_hot(train_y.astype(int), 10)
valid_y = one_hot(valid_y.astype(int), 10)
test_y = one_hot(test_y.astype(int), 10)


# Entrada (28x28)
x = tf.placeholder("float", [None,784])
y_ = tf.placeholder("float", [None,10])


W1 = tf.Variable(np.float32(np.random.rand(784,30)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(30)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(30,10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10))* 0.1)

h = tf.nn.sigmoid(tf.matmul(x,W1) + b1)

y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
errorAnterior = 1000000
errorActual = 0
epoch = 0
finish = True
check = 0
errorTabla = []


while finish:
    for jj in xrange(len(train_x) / batch_size):
        # Fase de entrenamiento
        # En cada vuelta vamos cogiendo 20 muestras hasta que termine el bucle
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Fase de validacion
    errorActual = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    errorTabla.append(errorActual)

    epoch = epoch + 1
    errorNuevo = 0
    if ( errorActual >= errorAnterior):
        check += 1
        if( check > 6 ):
            finish = False
            print "* Fin del entrenamiento *"
            print "-Error anterior: " + str(errorAnterior)
            print "-Error actual: " + str(errorActual)
            break
    else:
        check = 0
        errorNuevo = errorAnterior
        errorAnterior = errorActual


    print "Epoch:", epoch, "Error:", errorActual
    print "-Error anterior: " + str(errorNuevo)
    print "-Error actual: " + str(errorActual)
    print "----------------------------------------------------------------------------------"



print ""
print "¡La fase de entrenamiento ha terminado!"
print "----------------------------------------------------------------------------------"
print ""
print "Empezando la fase de test..."
print ""
time.sleep(10)

aciertos = 0

result = sess.run(y, feed_dict={x: test_x})

print "Test", "Error:", sess.run(loss, feed_dict={x: test_x, y_: test_y})

j = 0
for b, r in zip(test_y, result):
    if np.argmax(b) == np.argmax(r):
        aciertos += 1
    """else:
        # si da error imprimimos el numero para ver como es
        plt.imshow(test_x[j].reshape((28, 28)), cmap=cm.Greys_r)
        plt.show()  # Let's see a sample
        print test_y[j]
    j += 1
    
    """

print "Porcentaje de aciertos: " + str(aciertos/float(len(test_y))*100) + "%"
plt.plot(errorTabla)
plt.ylabel("Error")
plt.show()
print "----------------------------------------------------------------------------------"