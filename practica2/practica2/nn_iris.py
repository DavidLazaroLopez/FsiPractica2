# -*- coding: utf-8 -*-

from random import randint
import time
import tensorflow as tf
import numpy as np
#importamos tenso flow

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
# funcion que le pasas un 0 y devuelve 1,0,0,0
# n es el numero de bits
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

# Generamos una matriz a partir del fichero .data
data = np.genfromtxt('iris.data', delimiter=",")
# Ordenamos los datos de forma aleatoria para entrenar
np.random.shuffle(data)
x_data = data[:, 0:4].astype('f4')
y_data = one_hot(data[:, 4].astype(int), 3)

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# creamos una matriz de 4x5 con valores random entre 0 y 0,1
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
#  5 neuronas = 5 umbrales
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)
# matriz de 5x3
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
# 3 neuronas = 3 umbrales
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)


h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)


# tf.reduce_sum : calcula la suma de todos los elementos en una capa
loss = tf.reduce_sum(tf.square(y_ - y))


# Optimizamos la función para bajar el error
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

"""
print "----------------------"
print "      Apartado A      "
print "----------------------"

# Entrenar con todas las muestras X veces y comprobar los errores por ronda

# np.argmax() Coge la posicion del mayor valor de la lista
def errorDetect(b, r):
    if ( np.argmax(b) != np.argmax(r) ):
        return True

ciclosTest = 0
pos = 0
fallos = 0

for jj in xrange(1):
    batch_xs = x_data[ 0 : len(x_data) ] # x_data[0 : 150]
    batch_ys = y_data[ 0 : len(x_data) ] # y_data[0 : 150]
    # ejecuta train alimentado por batch_xs, batch_ys
    trainData = sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    ciclosTest += 1

actualError = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
print "Epoch:", ciclosTest, "Error:", actualError
result = sess.run(y, feed_dict={x: batch_xs})
muestra = 1
for b, r in zip(batch_ys, result):
    print muestra, ".-", b, "-->", r
    muestra += 1
    if( errorDetect(b, r) ):
        fallos += 1

print "Cantidad de errores (" + str(fallos) + "/150): " + str((fallos / float(150)) * 100) + "%"
print "----------------------------------------------------------------------------------"



"""
print "----------------------"
print "      Apartado B      "
print "----------------------"


batch_size = 20
trainingSamples = int(len(x_data) * 0.7) # 70% de las muestras (105(0-104))
ciclosValidacion = 0
finish = True
lastError = 100
check = 0;

while finish:
    for jj in xrange(10):
        randomInt = randint(0, trainingSamples - batch_size)
        batch_xs = x_data[ trainingSamples - batch_size - randomInt : trainingSamples - randomInt ]
        batch_ys = y_data[ trainingSamples - batch_size - randomInt : trainingSamples - randomInt ]
        trainData = sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Validacion
    ciclosValidacion += 1
    batch_xs = x_data[ trainingSamples + 1 : (len(x_data) - 1) - int(len(x_data) * 0.15) ] # x_data[106: 126]
    batch_ys = y_data[ trainingSamples + 1 : (len(x_data) - 1) - int(len(x_data) * 0.15) ] # y_data[106: 126]
    actualError = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})

    # Terminar con las pruebas de validacion
    # Solo terminamos cuando un error es superado por los errores de las 6 siguientes epocas
    if ( actualError >= lastError ):
        check += 1;
        if( check > 6 ):
            finish = False
            print "*Fin del entrenamiento*"
            print "-Error anterior: " + str(lastError)
            print "-Error actual: " + str(actualError)
            break
    else:
        check = 0
        lastError = actualError

    print "Epoch:", ciclosValidacion, "Error:", actualError
    print "-Error anterior: " + str(lastError)
    print "-Error actual: " + str(actualError)
    #result = sess.run(y, feed_dict={x: batch_xs})
    # [... (batch_ys[0], result[0]) , (batch_ys[1], result[1]) ...]
    # cuando alguna tupla de las pasadas por parametros se quede sin elementos, termina.
    #muestra = 1
    #for b, r in zip(batch_ys, result): # batch_ys es lo que tiene que dar [1,0,0] y result es el resultado del entrenamiento
     #   print muestra, ".-", b, "-->", r
      #  muestra += 1
    print "----------------------------------------------------------------------------------"


print ""
print "¡La fase de entrenamiento ha terminado!"
print "Empezando la fase de test..."
print ""
time.sleep(10)


ciclosTest = 0

batch_xs = x_data[ (len(x_data) - 1) - int(len(x_data) * 0.15) : len(x_data) ]
batch_ys = y_data[ (len(x_data) - 1) - int(len(x_data) * 0.15) : len(x_data) ]
ciclosTest += 1

actualError = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
print "Test:", ciclosTest, "Error:", actualError
result = sess.run(y, feed_dict={x: batch_xs})
muestra = 1
for b, r in zip(batch_ys, result):
    print muestra, ".-", b, "-->", r
    muestra += 1
print "----------------------------------------------------------------------------------"