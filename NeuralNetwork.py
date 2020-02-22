import numpy, sys 

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

debug = False

# activation functions and their derivatives
def logistic(x) : return 1 / (1 + numpy.exp(-x))
def logistic_deriv(x) : return logistic(x) * (1 - logistic(x))
def linear(x) : return x
def linear_deriv(x) : return 1

# online loss functions and their derivatives
def square_error(prediction, target):
    total = 0
    for i in range(len(prediction)):
        total += (prediction[i]-target[i])**2
    return total
def log_loss(prediction, target):
    total = 0
    for i in range(len(prediction)):
        if target == 1:
            total += -log(prediction)
        else:
            total += -log(1 - prediction)
    return total
def square_error_deriv(prediction, target):
    return prediction - target
def log_loss_deriv(prediction, target):
    return - (target/prediction) + ((1-target)/(1-prediction))

# batch loss functions
def mse(data):
    total = 0
    for sample in data:
        prediction, target = sample
        total += square_error(prediction,target)
    return total/len(data)

# get the derivative of a function
def get_deriv(function):
    if function == logistic: return logistic_deriv 
    elif function == linear: return linear_deriv
    elif function == square_error: return square_error_deriv
    elif function == log_loss: return log_loss_deriv
    else: return None

class Neuron:
    def __init__(self, activation, num_inputs, eta, weights):
        self.activation = activation
        self.num_inputs = num_inputs
        self.eta = eta
        self.weights = []
        if weights == "random":
            for i in range(num_inputs+1):
               self.weights.append(numpy.random.random())
        else:
            self.weights = weights

    def activate(self, x):
        return self.activation(x)

    def calculate(self, data):
        total = 0;
        for i in range(len(data)):
            total += data[i] * self.weights[i]

        self.prev = data
        self.net = total + self.weights[len(data)]
        self.out = self.activate(self.net)

        return self.out

    def train(self, deriv):
        delta = get_deriv(self.activation)(self.net) * deriv

        weight_deltas = []
        for i in range(len(self.prev)):
            weight_deltas.append(delta * self.weights[i])

        # update weights
        for i in range(len(self.prev)):
            self.weights[i] -= self.eta * delta * self.prev[i]
        self.weights[i+1] -= self.eta * delta

        return weight_deltas

    def print(self):
        print("num_inputs: " + str(self.num_inputs))
        print("eta: " + str(self.eta))
        count = 0
        for i in self.weights:
            print("weight[" + str(count) + "]: " + str(i))
            count += 1

class FullyConnectedLayer:
    def __init__(self, num_neurons, activation, num_inputs, eta, weights):
        self.activation = activation
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.eta = eta

        self.neurons = []
        if weights == "random":
            for i in range(self.num_neurons):
                self.neurons.append(Neuron(self.activation,
                                           self.num_inputs,
                                           self.eta,
                                           "random"))
        else:
            for i in range(self.num_neurons):
                self.neurons.append(Neuron(self.activation,
                                           self.num_inputs,
                                           self.eta,
                                           weights[i]))

    def calculate(self, data):
        output = []
        for i in range(self.num_neurons):
            output.append(self.neurons[i].calculate(data))
        return output

    def train(self, derivs):
        delta_sums = self.neurons[0].train(derivs[0])
        for i in range(1,self.num_neurons):
            delta_sums = numpy.add(delta_sums,
                                   self.neurons[i].train(derivs[i]))
        return(delta_sums)

    def print(self):
        for i in range(self.num_neurons):
            print("neuron " + str(i))
            self.neurons[i].print()
            print("")

class ConvolutionalLayer:
    def __init__(self, num_kernels, kernel_size, activation, input_dimension, eta, weights):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_dimension = input_dimension
        self.eta = eta
        self.weights = weights
        self.output_dimension = [input_dimension[0] - kernel_size + 1, input_dimension[1] - kernel_size + 1]
        print("size of output: " + str(self.output_dimension))
        self.kernels = []
        for k in range(num_kernels):
            rows = []
            for i in range(self.output_dimension[0]):
                column = []
                for j in range(self.output_dimension[1]):
                    column.append(Neuron(self.activation,
                                         numpy.prod(self.output_dimension),
                                         self.eta,
                                         self.weights[k]))
                rows.append(column)
            self.kernels.append(rows)

        if debug:
            print("kernels: ")
            for k in range(num_kernels):
                print("kernel #: " + str(k))
                for i in range(self.output_dimension[0]):
                    print("row " + str(i))
                    for j in range(self.output_dimension[1]):
                        print("col " + str(j))
                        self.kernels[k][i][j].print()

    def calculate(self, data):
        k_out = []
        for k in range(self.num_kernels):
            r_out = []
            for r_idx in range(self.output_dimension[0]):
                c_out = []
                for c_idx in range(self.output_dimension[1]):
                    total = 0
                    for i in range(self.output_dimension[0]):
                        for j in range(self.output_dimension[1]):
                            if debug:
                                print("c_idx: " + str(c_idx))
                                print("r_idx: " + str(r_idx))
                                print("i: " + str(i))
                                print("j: " + str(j))
                            total += data[r_idx+i][c_idx+j] * self.kernels[k][i][j].weights[self.output_dimension[1]*i+j]
                    c_out.append(total)
                r_out.append(c_out)
                if debug: print()
            k_out.append(r_out)
    
        #print(k_out)
        return k_out

class FlattenLayer:
    def __init__(self, input_size):
        self.input_size = input_size

    def calculate(self, data):
        #print(data)
        return numpy.asarray(data).flatten().tolist()

class MaxPoolingLayer:
    def __init__(self, kernel_size, input_dimension):
        self.kernel_size = kernel_size
        self.input_dimension = input_dimension

    def calculate(self, data):
        k_out = []
        for k in range(self.input_dimension[0]):
            r_out = []
            for r_idx in range(0,self.input_dimension[1],self.kernel_size):
                c_out = []
                for c_idx in range(0,self.input_dimension[2],self.kernel_size):
                    pool = []
                    for i in range(self.kernel_size):
                        for j in range(self.kernel_size):
                            if debug:
                                print("k: " + str(k))
                                print("c_idx: " + str(c_idx))
                                print("r_idx: " + str(r_idx))
                                print("i: " + str(i))
                                print("j: " + str(j))
                                print(data[k][r_idx+i][c_idx+j])
                            pool.append(data[k][r_idx+i][c_idx+j])
                    c_out.append(max(pool))
                r_out.append(c_out)
            k_out.append(r_out)
        
        #print(k_out)
        return k_out

class NeuralNetwork:
    def __init__(self, input_size, loss, eta):
        self.loss = loss
        self.eta = eta
        self.layers = []

    def addLayer(self, layer, **kwargs):
        if debug:
            print("adding new layer of type:")
            print(layer)
        newLayer = layer(**kwargs)
        self.layers.append(newLayer)

        #if not len(self.layers):
            #self.layers.append(FullyConnectedLayer(  


    def calculate(self, data):
        output = data
        for layer in self.layers:
            print("calculating output for layer: ", end='')
            print(layer)
            output = layer.calculate(output)
            print("output: " + str(output), end='\n\n')
        return output

    def calculate_loss(self, sample, target):
        return self.loss(self.calculate(sample), target)

    def train(self, data, target):
        prediction = self.calculate(data)

        derivs = []
        for i in range(len(prediction)):
            derivs.append(get_deriv(self.loss)(prediction[i],target[i]))

        for i in range(self.num_layers-1,-1,-1):
            derivs = self.layers[i].train(derivs)

    def print(self):
        for i in range(self.num_layers):
            print("Layer " + str(i))
            print("Number of Neurons in Layer: " + 
                  str(self.layers[i].num_neurons))
            self.layers[i].print()

def main():
    choices = ['example1', 'example2', 'example3']
    if len(sys.argv) != 2 or sys.argv[1] not in choices:
        print("Please provide one of the following command line arguments:")
        print(choices)
        exit()

    if sys.argv[1] == 'example1':
        print("Running example1.")

        model=Sequential()

        # add convolutional layer
        model.add(layers.Conv2D(filters=1,
                                kernel_size=3,
                                strides=1,
                                padding='valid',
                                input_shape=(5,5,1)))

        weight = [numpy.asarray([[[[1]],[[0]],[[1]]],
                                 [[[0]],[[1]],[[0]]],
                                 [[[1]],[[0]],[[1]]]]),
                  numpy.asarray([0])]

        model.layers[0].set_weights(weight) 
        
        # add a flatten layer
        model.add(layers.Flatten())

        # add fully connected layer
        model.add(layers.Dense(1, activation='sigmoid'))

        weight = [numpy.asarray([[-0.1],
                                 [ 0.2],
                                 [-0.3],
                                 [ 0.4],
                                 [-0.5],
                                 [ 0.6],
                                 [-0.7],
                                 [ 0.8],
                                 [-0.9]]),
                  numpy.array([0])]

        model.layers[2].set_weights(weight)

        print('Keras Kernel Weights:')
        print(model.layers[0].get_weights(), end='\n\n')

        print('Keras Fully Connected Weights:')
        print(model.layers[2].get_weights(), end='\n\n')

        # prepare model for training
        sgd = keras.optimizers.SGD(learning_rate=0.1,
                                   momentum=0.0,
                                   nesterov=False)
        model.compile(loss='mean_squared_error', optimizer=sgd)

        image = numpy.asarray([[[[1],[1],[1],[0],[0]],
                                [[0],[1],[1],[1],[0]],
                                [[0],[0],[1],[1],[1]],
                                [[0],[0],[1],[1],[0]],
                                [[0],[1],[1],[0],[0]]]])

        print('Input Image:')
        print(image, end='\n\n')

        print('Keras Model Output')
        print(model.predict(image), end='\n\n')

        print('Keras Model Evaluation')
        model.evaluate(image,numpy.asarray([[1]]))
        print()

#        model.fit(image, numpy.asarray([[1]]), epochs=1, verbose=0)
#
#        print('Keras Updated Kernel Weights:')
#        print(model.layers[0].get_weights(), end='\n\n')
#
#        print('Keras Updated Fully Connected Weights:')
#        print(model.layers[2].get_weights(), end='\n\n')
#
#        print('Keras Updated Model Evaluation')
#        model.evaluate(image,numpy.asarray([[1]]))

        
        print("My cnn")
        cnn = NeuralNetwork(input_size = 5,
                            loss = square_error,
                            eta = .1)

        weights = [[1,0,1,0,1,0,1,0,1]]

        # add convolutional layer
        cnn.addLayer(ConvolutionalLayer, num_kernels=1, kernel_size=3,
                     activation=logistic, input_dimension=(5,5), eta=.1,
                     weights=weights)

        # add flatten layer
        cnn.addLayer(FlattenLayer, input_size=[1,3,3])

        # add fully connected layer
        weights = [[-0.1,0.2,-0.3,0.4,-0.5,0.6,-0.7,0.8,-0.9,0.0]]

        cnn.addLayer(FullyConnectedLayer, 
                     num_neurons = 1,
                     activation = logistic,
                     num_inputs = 9,
                     eta = .1,
                     weights = weights)

        image = [[1,1,1,0,0],
                 [0,1,1,1,0],
                 [0,0,1,1,1],
                 [0,0,1,1,0],
                 [0,1,1,0,0]]

        print("My Model Output")
        print(cnn.calculate(image))

        print("My Model Evaluation")
        print(cnn.calculate_loss(image,[1]))

        #result = cnn.layers[1].calculate(result)
        #print(result)
        #cnn.addLayer(MaxPoolingLayer, kernel_size=2,input_dimension=[1,4,4])
        #result = [[[12,20,30,0],[8,12,2,0],[34,70,37,4],[112,100,25,12]]]
        #cnn.layers[1].calculate(result)

        exit()

        num_inputs = 2
        num_layers = 2
        num_neurons = [2,2]
        weights = [[[.15,.20,.35],[.25,.30,.35]],
                   [[.40,.45,.60],[.50,.55,.60]]]

        nn = NeuralNetwork(num_layers, num_neurons, logistic, 
                           num_inputs, square_error, .5, weights)

        print("Loss before training example: ", end = '')
        print(nn.calculate_loss([.05,.10],[.01,.99]))

        nn.train([.05,.10],[.01,.99])

        print("Loss after training example: ", end = '')
        print(nn.calculate_loss(nn.calculate([.05,.10]),[.01,.99]))

    elif sys.argv[1] == 'and':
        print("Running 'and' example.")
        num_inputs = 2
        num_layers = 2
        num_neurons = [1,1]
        weights = "random"

        nn = NeuralNetwork(num_layers, num_neurons, logistic, 
                           num_inputs, square_error, .1, weights)

        inputs = [([0,0],[0]), ([0,1],[0]), 
                  ([1,0],[0]), ([1,1],[1])]

        for i in range(8000):
            for sample, target in inputs:
                nn.train(sample, target)

        print("Outputs for all 4 inputs after training.")
        print("0 and 0 -> " + str(nn.calculate([0,0])))
        print("1 and 0 -> " + str(nn.calculate([1,0])))
        print("0 and 1 -> " + str(nn.calculate([0,1])))
        print("1 and 1 -> " + str(nn.calculate([1,1])))
        
    else:
        print("Running 'xor' example.")
        num_inputs = 2
        num_layers = 2
        num_neurons = [1,1]
        weights = "random"

        print("Training with one perceptron.")
        inputs = [([0,0],[0]), ([0,1],[1]),
                  ([1,0],[1]), ([1,1],[0])]

        nn = NeuralNetwork(num_layers, num_neurons, logistic, 
                           num_inputs, square_error, .2, weights)

        for i in range(8000):
            for sample, target in inputs:
                nn.train(sample, target)

        print("Outputs for all 4 inputs after training.")
        print("0 and 0 -> " + str(nn.calculate([0,0])))
        print("1 and 0 -> " + str(nn.calculate([1,0])))
        print("0 and 1 -> " + str(nn.calculate([0,1])))
        print("1 and 1 -> " + str(nn.calculate([1,1])))
        print()

        num_inputs = 2
        num_layers = 2
        num_neurons = [2,1]
        weights = "random"

        print("Training with multiple perceptrons.")
        inputs = [([0,0],[0]), ([0,1],[1]),
                  ([1,0],[1]), ([1,1],[0])]

        nn = NeuralNetwork(num_layers, num_neurons, logistic, 
                           num_inputs, square_error, .2, weights)

        for i in range(8000):
            for sample, target in inputs:
                nn.train(sample, target)

        print("Outputs for all 4 inputs after training.")
        print("0 and 0 -> " + str(nn.calculate([0,0])))
        print("1 and 0 -> " + str(nn.calculate([1,0])))
        print("0 and 1 -> " + str(nn.calculate([0,1])))
        print("1 and 1 -> " + str(nn.calculate([1,1])))

if __name__ == '__main__':
    main()

