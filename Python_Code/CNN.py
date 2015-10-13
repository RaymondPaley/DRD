import re
import numpy, theano, sys, math
from theano import tensor as T
from theano import shared
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
 
BATCH_SIZE=200
 
def relu_f(vec):
    """ Wrapper to quickly change the rectified linear unit function """
    return (vec + abs(vec)) / 2.
         
def dropout(rng, x, p=0.5):
    """ Zero-out random values in x with probability p using rng """
    if p > 0. and p < 1.:
        seed = rng.randint(2 ** 30)
        srng = theano.tensor.shared_randomstreams.RandomStreams(seed)
        mask = srng.binomial(n=1, p=1.-p, size=x.shape,
                dtype='float32') #change
        return T.cast(x * mask,'float32')
    return  T.cast(x,'float32')
         
def fast_dropout(rng, x):
    """ Multiply activations by N(1,1) """
    seed = rng.randint(2 ** 30)
    srng = RandomStreams(seed)
    mask = srng.normal(size=x.shape, avg=1., dtype='float32') #change
    return T.cast(x * mask,'float32')
         
def build_shared_zeros(shape, name):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return shared(value=numpy.zeros(shape, dtype='float32'), #change
            name=name, borrow=True)
         
 
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """
 
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
            filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                                image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
 
        assert image_shape[1] == filter_shape[1]
        self.input = input
 
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype='float32'),
            borrow=True)
 
        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype='float32')
        self.b = theano.shared(value=b_values, borrow=True)
 
        # convolve input feature maps with filters
        conv_out = T.cast(conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape),'float32')
 
        # downsample each feature map individually, using maxpooling
        pooled_out = T.cast(downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True),'float32')
 
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + T.cast(self.b.dimshuffle('x', 0, 'x', 'x'),'float32'))
        # TODO relu output of convolutions
        #self.output = relu_f(pooled_out + T.cast(self.b.dimshuffle('x', 0, 'x', 'x'),'float32')) 
        
        # store parameters of this layer
        self.params = [self.W, self.b]
    def __repr__(self):
        return "ConvPoolLayer" #might have to change this
 
 
 
class ReLU(object):
    """ Basic rectified-linear transformation layer (W.X + b) 
        Multipurpose"""
    def __init__(self, rng, input, n_in, n_out, drop_out=0.0, W=None, b=None, fdrop=False):
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype='float32')
            W_values *= 4  
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b = build_shared_zeros((n_out,), 'b')
        self.input = input
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        self.output = T.dot(self.input, self.W) + self.b
        self.pre_activation = self.output
        if drop_out:
            if fdrop:
                self.pre_activation = fast_dropout(rng, self.pre_activation)
                self.output = relu_f(self.pre_activation) 
            else:
                self.W=W * 1. / (1. - dr)   
                self.b=b * 1. / (1. - dr)
                self.output = dropout(numpy_rng, self.output, dr)
                self.output = relu_f(self.pre_activation) 
        else:
            self.output = relu_f(self.pre_activation) 
    def __repr__(self):
        return "ReLU"
 
class DatasetMiniBatchIterator(object):
    def __init__(self, x, y, batch_size=200, randomize=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.randomize = randomize
        from sklearn.utils import check_random_state
        self.rng = check_random_state(42)
 
    def __iter__(self):
        n_samples = self.x.shape[0]
        if self.randomize:
            for _ in xrange(n_samples / BATCH_SIZE):
                if BATCH_SIZE > 1:
                    i = int(self.rng.rand(1) * ((n_samples+BATCH_SIZE-1) / BATCH_SIZE))
                else:
                    i = int(math.floor(self.rng.rand(1) * n_samples))
                yield (i, self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])
        else:
            for i in xrange((n_samples + self.batch_size - 1)
                            / self.batch_size):
                yield (self.x[i*self.batch_size:(i+1)*self.batch_size],self.y[i*self.batch_size:(i+1)*self.batch_size])
 
class LogisticRegression:
    """Multi-class Logistic Regression (no dropout in this layer)
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        if W != None:
            self.W = W
        else:
            self.W = build_shared_zeros((n_in, n_out), 'W')
        if b != None:
            self.b = b
        else:
            self.b = build_shared_zeros((n_out,), 'b')
 
        # P(Y|X) = softmax(W.X + b)
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #this is the prediction. pred
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.output = self.y_pred
        self.params = [self.W, self.b]
 
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
 
    def negative_log_likelihood_sum(self, y):
        return -T.sum(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
 
    def training_cost(self, y):
        """ Wrapper for standard name """
        return self.negative_log_likelihood_sum(y)
 
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                ("y", y.type, "y_pred", self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            print("!!! y should be of int type")
            return T.mean(T.neq(self.y_pred, numpy.asarray(y, dtype='int')))
 
    def prediction(self, input):
        return self.y_pred
 
 
class ConvDropNet(object):
    """ Convolutional Neural network class 
        Given the parameters for each problem. This class is left to be more customizeable
        and almost acts as function


    """
    def __init__(self, numpy_rng, theano_rng=None, 
                 n_ins=40*3,
                 conv_reshaper=(BATCH_SIZE, 1, 28, 28),
                 batch_size=BATCH_SIZE,
                 Conv ={'image_shape1':(BATCH_SIZE, 1, 28, 28),'image_shape2':(BATCH_SIZE, 20, 12, 12)},
                 filters={'filter_shape1':(20, 1, 5, 5),'filter_shape2':(50, 20, 5, 5)},
                 poolsize=(2,2),
                 layers_types=[LeNetConvPoolLayer,LeNetConvPoolLayer, ReLU, ReLU, ReLU, LogisticRegression],
                 layers_sizes=['NA', 'NA', 800, 500, 500], 
                 n_outs=62 * 3, 
                 rho=0.98,
                 eps=1.E-6,
                 max_norm=0.,
                 debugprint=True,
                 fast_drop=True,
                 dropout_rates=[0.,0., 0.5, 0.5, 0.5, 0.] #match this up with actual layers
                 ):
 
 
        self.layers = []
        self.params = []
        self.n_layers = len(layers_types)
        self.layers_types = layers_types
        assert self.n_layers > 0
        self.max_norm = max_norm
        self._rho = rho  # ``momentum'' for adadelta
        self._eps = eps  # epsilon for adadelta
        self._accugrads = []  # for adadelta
        self._accudeltas = []  # for adadelta
        if theano_rng == None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
 
        self.x = T.fmatrix('x')
        self.y = T.ivector('y')
        
        self.layers_ins = [n_ins] + layers_sizes
        self.layers_outs = layers_sizes + [n_outs]
        
        layer_input = self.x
 
 
        self.batch_size = BATCH_SIZE
 
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        conv_layer_input=T.cast(layer_input.reshape(conv_reshaper),'float32') #change later params
 
 
 
        
        
        #change these for each conv layer, and specify params
        self.poolsize=poolsize
        
 
        self.dropout_rates = dropout_rates
        if fast_drop:
            if dropout_rates[0]:
                dropout_layer_input = fast_dropout(numpy_rng, self.x)
            else:
                dropout_layer_input = self.x
        else:
            dropout_layer_input = dropout(numpy_rng, self.x, p=dropout_rates[0])
        self.dropout_layers = []
 
 
        layer0=LeNetConvPoolLayer(rng=numpy_rng, input=conv_layer_input, 
          filter_shape=filters['filter_shape1'], image_shape=Conv['image_shape1'], 
          poolsize=self.poolsize)
        self.params.extend(layer0.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer0.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer0.params])
        assert hasattr(layer0, 'output')   
        self.dropout_layers.append(layer0)
        dropout_layer_input = T.cast(layer0.output,'float32')
        #print dropout_layer_input
 
        layer1=LeNetConvPoolLayer(rng=numpy_rng,input=dropout_layer_input, filter_shape=filters['filter_shape2'], 
          image_shape=Conv['image_shape2'], poolsize=self.poolsize)
        self.params.extend(layer1.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
           'accugrad') for t in layer1.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
           'accudelta') for t in layer1.params])
        assert hasattr(layer1, 'output')
        self.dropout_layers.append(layer1)
        dropout_layer_input = T.cast(layer1.output.flatten(2),'float32')
        #print dropout_layer_input
        
        # construct fully-connected ReLU layers
        layer2= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[2] ,fdrop=True, n_in=50 * 4 * 4, n_out=500)
        self.params.extend(layer2.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer2.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer2.params])
        assert hasattr(layer2, 'output')
        self.dropout_layers.append(layer2)
        dropout_layer_input = layer2.output
        #print dropout_layer_input   
 
        layer3= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[3] , fdrop=True, n_in=500, n_out=500)    
        self.params.extend(layer3.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer3.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer3.params])
        assert hasattr(layer3, 'output')
        self.dropout_layers.append(layer3)
        dropout_layer_input = T.cast(layer3.output,'float32')
        #print dropout_layer_input
 
        layer4= ReLU(rng=numpy_rng, input=dropout_layer_input, drop_out=dropout_rates[4] , fdrop=True, n_in=500, n_out=500)
        self.params.extend(layer4.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer4.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer4.params])
        assert hasattr(layer4, 'output')
        self.dropout_layers.append(layer4)
        dropout_layer_input = T.cast(layer4.output,'float32')
        #print dropout_layer_input
 
        # classify the values
        layer5= LogisticRegression(rng=numpy_rng, input=dropout_layer_input, n_in=500, n_out=10)
        self.params.extend(layer5.params)
        self._accugrads.extend([build_shared_zeros(t.shape.eval(),
            'accugrad') for t in layer5.params])
        self._accudeltas.extend([build_shared_zeros(t.shape.eval(),
            'accudelta') for t in layer5.params])
        assert hasattr(layer5, 'output')
        self.dropout_layers.append(layer5)
        dropout_layer_input = T.cast(layer5.output,'float32')
        print dropout_layer_input
 
 
        assert hasattr(self.dropout_layers[-1], 'training_cost')
        assert hasattr(self.dropout_layers[-1], 'errors')
 
        self.mean_cost = self.dropout_layers[-1].negative_log_likelihood(self.y)
        #print self.mean_cost.dtype
        self.cost = self.dropout_layers[-1].training_cost(self.y)
        self.prediction_1 = self.dropout_layers[-1].prediction(self.x)
        if debugprint:
            theano.printing.debugprint(self.cost) 
        # the non-dropout errors
        self.errors = self.dropout_layers[-1].errors(self.y)
 
    
    def __repr__(self):
        dimensions_layers_str = map(lambda x: "x".join(map(str, x)),
                                    zip(self.layers_ins, self.layers_outs))
        return "_".join(map(lambda x: "_".join((x[0].__name__, x[1])),
                            zip(self.layers_types, dimensions_layers_str))) + "\n"\
        + "dropout rates: " + str(self.dropout_rates)
 
 
    def get_SGD_trainer(self):
        """ Returns a plain SGD minibatch trainer with learning rate as param.
        """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        # using mean_cost so that the learning rate is not too dependent
        # on the batch size
        gparams = T.grad(self.mean_cost, self.params)
 
        # compute list of weights updates
        updates = OrderedDict()
        for param, gparam in zip(self.params, gparams):
            if self.max_norm:
                W = param - gparam * learning_rate
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param - gparam * learning_rate
 
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y),
                                           theano.Param(learning_rate)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y})
 
        return train_fn
 
    def get_adagrad_trainer(self):
        """ Returns an Adagrad (Duchi et al. 2010) trainer using a learning rate.
        """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        learning_rate = T.fscalar('lr')  # learning rate to use
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
 
        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(self._accugrads, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            dx = - (learning_rate / T.sqrt(agrad + self._eps)) * gparam
            if self.max_norm:
                W = param + dx
                col_norms = W.norm(2, axis=0)
                desired_norms = T.clip(col_norms, 0, self.max_norm)
                updates[param] = W * (desired_norms / (1e-6 + col_norms))
            else:
                updates[param] = param + dx
            updates[accugrad] = agrad
 
        train_fn = theano.function(inputs=[theano.Param(batch_x), 
            theano.Param(batch_y),
            theano.Param(learning_rate)],
            outputs=self.mean_cost,
            updates=updates,
            givens={self.x: batch_x, self.y: batch_y})
 
        return train_fn
 
    def get_adadelta_trainer(self):
        """ Returns an Adadelta (Zeiler 2012) trainer using self._rho and
        self._eps params.
        """
        batch_x = T.fmatrix('batch_x')
        batch_y = T.ivector('batch_y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.mean_cost, self.params)
        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, accudelta, param, gparam in zip(self._accugrads,
                self._accudeltas, self.params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = self._rho * accugrad + (1 - self._rho) * gparam * gparam
            agrad=T.cast(agrad, 'float32')
            dx = - T.sqrt((accudelta + self._eps)
                          / (agrad + self._eps)) * gparam
            dx=T.cast(dx, 'float32')
            updates[accudelta] = T.cast((self._rho * accudelta
                                  + (1 - self._rho) * dx * dx),'float32')
            if self.max_norm:
                W = T.cast(param + dx,'float32')
                col_norms = T.cast(W.norm(2, axis=0),'float32')
                desired_norms = T.cast(T.clip(col_norms, 0, self.max_norm),'float32')
                updates[param] = T.cast(W * (desired_norms / (1e-6 + col_norms)),'float32')
            else:
                updates[param] = T.cast(param + dx,'float32')
            updates[accugrad] = T.cast(agrad,'float32')
 
        train_fn = theano.function(inputs=[theano.Param(batch_x),
                                           theano.Param(batch_y)],
                                   outputs=self.mean_cost,
                                   updates=updates,
                                   givens={self.x: batch_x, self.y: batch_y},
                                   allow_input_downcast=True)
 
        return train_fn
 
 
    def score_classif(self, given_set):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        batch_y = T.ivector('batch_y')
        score = theano.function(inputs=[theano.Param(batch_x),
                                        theano.Param(batch_y)],
                                outputs=self.errors,
                                givens={self.x: batch_x, self.y: batch_y},
                                allow_input_downcast=True)
 
        def scoref():
            """ returned function that scans the entire set given as input """
            return [score(batch_x, batch_y) for batch_x, batch_y in given_set]
 
        return scoref
 
    def predict(self,X):
        """ Returns functions to get current classification errors. """
        batch_x = T.fmatrix('batch_x')
        #batch_x= T.matrix('batch_x', dtype=theano.config.floatX)
        predictor = theano.function(inputs=[theano.Param(batch_x)],
                                outputs=self.prediction_1,
                                givens={self.x: batch_x},
                                allow_input_downcast=True)
        return predictor(X)
 
 
 
def add_fit_and_score_early_stop(class_to_chg):
    """ Mutates a class to add the fit() and score() functions to a NeuralNet.
    """
    from types import MethodType
    def fit(self, x_train, y_train, x_dev=None, y_dev=None,
            max_epochs=300, early_stopping=True, split_ratio=0.1,
            method='adadelta', verbose=False, plot=False):
 
        """
        Fits the neural network to `x_train` and `y_train`. 
        If x_dev nor y_dev are not given, it will do a `split_ratio` cross-
        validation split on `x_train` and `y_train` (for early stopping).
        """
        import time, copy
        if x_dev == None or y_dev == None:
            from sklearn.cross_validation import train_test_split
            x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                    test_size=split_ratio, random_state=42)
        if method == 'sgd':
            train_fn = self.get_SGD_trainer()
        elif method == 'adagrad':
            train_fn = self.get_adagrad_trainer()
        elif method == 'adadelta':
            train_fn = self.get_adadelta_trainer()
        train_set_iterator = DatasetMiniBatchIterator(x_train, y_train)
        dev_set_iterator = DatasetMiniBatchIterator(x_dev, y_dev)
        train_scoref = self.score_classif(train_set_iterator)
        dev_scoref = self.score_classif(dev_set_iterator)
        best_dev_loss = numpy.inf
        epoch = 0
 
        patience = 1000  # look as this many examples regardless 
        patience_increase = 2.  # wait this much longer when a new best is
                                    # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                           # considered significant
 
        done_looping = False
        print '... training the model'
        # early-stopping parameters
        test_score = 0.
        start_time = time.clock()
 
        done_looping = False
        epoch = 0
        timer = None
 
        if plot:
            verbose = True
            self._costs = []
            self._train_errors = []
            self._dev_errors = []
            self._updates = []
     
        while (epoch < max_epochs) and (not done_looping):
            epoch += 1
            if not verbose:
                sys.stdout.write("\r%0.2f%%" % (epoch * 100./ max_epochs))
                sys.stdout.flush()
            avg_costs = []
            timer = time.time()
            for iteration, (x, y) in enumerate(train_set_iterator):
                if method == 'sgd' or method == 'adagrad':
                    avg_cost = train_fn(x, y, lr=.04)  # LR is very dataset dependent
                elif method == 'adadelta':
                    avg_cost = train_fn(x, y)
                if type(avg_cost) == list:
                    avg_costs.append(avg_cost[0])
                else:
                    avg_costs.append(avg_cost)
                if patience <= iteration:  #i think i fixed this part
                    done_looping = True
                    break
            if verbose:
                mean_costs = numpy.mean(avg_costs)
                mean_train_errors = numpy.mean(train_scoref())
                print('  epoch %i took %f seconds' %
                      (epoch, time.time() - timer))
                print('  epoch %i, avg costs %f' %
                      (epoch, mean_costs))
                print('  epoch %i, training error %f' %
                      (epoch, mean_train_errors))
                if plot:
                    self._costs.append(mean_costs)
                    self._train_errors.append(mean_train_errors)
            dev_errors = numpy.mean(dev_scoref())
            if plot:
                self._dev_errors.append(dev_errors)
            if dev_errors < best_dev_loss:
                best_dev_loss = dev_errors
                best_params = copy.deepcopy(self.params)
                if verbose:
                    print('!!!  epoch %i, validation error of best model %f' %
                          (epoch, dev_errors))
                if (dev_errors < best_dev_loss *
                improvement_threshold):
                    patience = max(patience, iteration * patience_increase)
 
        if not verbose:
            print("")
        for i, param in enumerate(best_params):
            self.params[i] = param
     
    def score(self, x, y):
        """ error rates """
        iterator = DatasetMiniBatchIterator(x, y)
        scoref = self.score_classif(iterator)
        return numpy.mean(scoref())
 
     
    class_to_chg.fit = MethodType(fit, None, class_to_chg)
    class_to_chg.score = MethodType(score, None, class_to_chg)
 
 
if __name__ == "__main__":
    from sklearn import cross_validation, preprocessing
#    from sklearn.datasets import fetch_mldata
#    mnist = fetch_mldata('MNIST original')
#    X = numpy.asarray(mnist.data, dtype='float32')
#    y = numpy.asarray(mnist.target, dtype='int32')
    
    import os
    import glob
    import csv
    from skimage.io import imread
    import skimage.transform

    
#    os.chdir('/Users/tiruviluamala/Desktop/DRD')
    
   
    names = glob.glob("train/*")[0:100]
    
    y = [None] * 100
    
    
    for index, name in enumerate(names):
        f = open('trainLabels.csv')
        csv_f = csv.reader(f)    
        for row in csv_f:
            if row[0] == names[index].split('/')[1].split('.')[0]:
                y[index] = row[1]
                
    y = numpy.asarray(y, dtype='int32')
    
    X = [None] * 100    
    for index, name in enumerate(names):
        im = imread(names[index], as_grey=True)

        image = skimage.transform.downscale_local_mean(im,(int(im.shape[0]/30),int(im.shape[1]/30)))
        image = image[1:29,1:29]
        image = image.reshape((784))
        X[index] = image
    
    X = numpy.asarray(X, dtype = 'float32')
   
        
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(
                      X, y, test_size=0.2, random_state=42)
   
    add_fit_and_score_early_stop(ConvDropNet)
 
    dnn = ConvDropNet(numpy_rng=numpy.random.RandomState(123),
        theano_rng=None,   
        n_ins=x_train.shape[1],
        conv_reshaper=(BATCH_SIZE, 1, 28, 28),
        batch_size=BATCH_SIZE,
        Conv = {'image_shape1':(BATCH_SIZE, 1, 28, 28),'image_shape2':(BATCH_SIZE, 20, 12, 12)},
        filters={'filter_shape1':(20, 1, 5, 5),'filter_shape2':(50, 20, 5, 5)},
        poolsize=(2,2),
        layers_types=[LeNetConvPoolLayer,LeNetConvPoolLayer, ReLU, ReLU, ReLU, LogisticRegression],
        layers_sizes=['NA', 'NA', 800, 500, 500],
        n_outs=len(set(y_train)),
        rho=0.98,
        eps=1.E-6,
        max_norm=4,
        fast_drop=True, 
        dropout_rates=[0.,0., 0.5, 0.5, 0.5, 0.], #match this up with actual layers, last 
        debugprint=False)
        
    #train the model here, plot=False ,adadelta=fast and no learning rate need be provided
    dnn.fit(x_train, y_train, max_epochs=60, method='adadelta', verbose=True, plot=False) 
    
    #Determine answers for test set and format these answers into a submittable CSV file
    test_error = dnn.score(x_test, y_test)
    print("score: %f" % (1. - test_error))
 
    #predict function
    #import pickle 
    #pickle.dump(dnn,open('dnn.p','wb'))
    #test_results=dnn.predict((samples,features))