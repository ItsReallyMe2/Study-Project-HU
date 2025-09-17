import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.decomposition import PCA

class circular_autoencoder:
    def __init__(self, input_width):       
        """
        Input Layer
        """
        input_layer = keras.layers.Input(shape=(input_width,), name='input_layer')

        initializer = keras.initializers.random_normal(seed=0)
        
        """
        Encoder is a densely-connected layer with weight matrices (kernels) initialised according to the
        chosen kernel initializer, and initial biases of zero.
        """

        encoder = keras.layers.Dense(name='encoder',
                                 units=2,
                                 kernel_initializer=initializer,
                                 bias_initializer=initializer,
                                 )(input_layer)
        
        
        """
        Bottleneck layer enforcing the circular constraint and embedding is its output
        """
        embedding = circular_node()(encoder)
        
        """
        Output layer
        """
        output_layer = keras.layers.Dense(name='output_layer',
                                   units=input_width,
                                   kernel_initializer=initializer,
                                   bias_initializer=initializer
                                   )(embedding)

        self.model = keras.Model(outputs=output_layer, inputs=input_layer, name='circular_autoencoder')
        
    class loss_printing_callback(keras.callbacks.Callback):
        """
        Class to print final training loss and the the number of epochs needed for convergence
        """
        def __init__(self):
            super().__init__()
        
        def on_epoch_end(self, epoch, logs=None):
            self.epoch = epoch
            self.loss = logs['loss']
        
        def on_train_end(self, logs=None):
            print("Training ended on {0} epochs with a loss of {1:.4f}".format(self.epoch, self.loss))
                
    def train(self, data, batch_size=10, epochs=200, rate=0.3, momentum=0.5):
        """
        Train the model. It will not reset the weights each time so it can be called iteratively.
        data: data used for training
        batch_size: batch size for training, if unspecified default to 32 as is set by keras
        epochs: number of epochs in training
        rate: training rate
        """
        opt = tf.keras.optimizers.legacy.SGD(learning_rate=rate, momentum=momentum)

        self.model.compile(loss='mean_squared_error',
                           optimizer=opt)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss')

        opt_monitor = self.model.fit(data, data, batch_size=batch_size, epochs=epochs, 
                                     verbose=0, callbacks=early_stopping, shuffle=True)

        return opt_monitor
        
    def predict_phase(self, data):
        """
        Calculates the predicted phases (in radians) for the samples in the data provided using the trained autoencoder network
        """
        projection = self.predict_circnode_output(data)
        return (np.arctan2(projection[:, 0], projection[:, 1])/(2*np.pi)) % 1

    def predict_circnode_output(self, data):
        """
        Computes the bottleneck nodes output (the embedding) for the provided input data.
        """
        projection = keras.backend.function(inputs=[self.model.get_layer('input_layer').input],
                                     outputs=[self.model.get_layer('embedding').output]
                                     )([data])
        return projection[0]

class cyclops(circular_autoencoder):
    """
    This class is to be initialized with a numpy matrix with features along columns
    """
    def __init__(self, data, frac_var=0.85, dfrac_var=0.03):
        if not isinstance(data, np.ndarray):
            raise TypeError

        self.data = data

        """ Retain only PCs that capture 85% of the variance in the data """
        pca = PCA()
        transformed_data = pca.fit_transform(data)
        n_pcs = (np.cumsum(pca.explained_variance_ratio_)>frac_var).nonzero()[0][0]
        n_pcs = np.minimum(n_pcs, np.sum(pca.explained_variance_ratio_>dfrac_var))
        n_pcs = max(n_pcs, 2)
        reduced_data = transformed_data[:, slice(0, n_pcs)]
        self.pca = pca
        self.n_pcs = n_pcs

        """ Keras likes inputs in the [-1,1] range, so we scale the data to be in that range """
        rng = np.max(np.abs(reduced_data.flatten()))
        self.rng = rng
        self.reduced_data = reduced_data/rng
        self.linear_rss = np.sum(np.var(self.reduced_data[:,1:], axis=0) * self.reduced_data.shape[0])

        self.input_width = reduced_data.shape[-1]

        """ Initialize the Keras based autoencoder module """
        super().__init__(self.input_width)
    
    def train_model(self, best_of=40, batch_size=10, epochs=200, rate=0.3, momentum=0.5):
        """
        Trains 'best_of' number of autoencoders with training parameters and choses the one with lowest loss
        """
        candidates = [circular_autoencoder(self.input_width).train(self.reduced_data, batch_size=batch_size, epochs=epochs, rate=rate, momentum=momentum) for i in np.arange(best_of)]

        self.training_losses = np.array([c.history['loss'][-1] for c in candidates])

        best_candidate_index = np.argmin(self.training_losses)
        self.model.set_weights(candidates[best_candidate_index].model.get_weights())

    def predict_phase(self, data=None):
        """
        Assigns phase values (in radians) to all the samples (rows) in the input data using the trained network. If no data is provided the training data is used.
        """
        if data is None:
            data = self.reduced_data
        else:
            transformed_data = self.pca.transform(data)
            reduced_data = transformed_data[:, slice(0, self.n_pcs)]
            data = reduced_data/self.rng
        return super().predict_phase(data)
        
    def feature_impute(self, x, blunt=0.95):
        """
        Replaces the any values outside center 'percent' of the data with (1-percent/2) and (percent/2) values appropriately.
        """
        lower_bound, upper_bound = np.percentile(x, [(1-blunt)*100/2, (1 + blunt)*100/2])
        return np.maximum(np.minimum(x, upper_bound), lower_bound)
    
    def calculate_metrics(self):
        """
        Calculates the two metrics: stats and smoothness described in Anafi et al.
        """
        circ_error = self.model.predict(self.reduced_data) - self.reduced_data
        self.circ_rss = np.sum(circ_error.flatten()**2)
        self.stat_err = (self.linear_rss - self.circ_rss)/self.circ_rss

        circ_ordering = np.argsort(self.predict_phase())
        linear_ordering = np.argsort(self.reduced_data)

        self.linear_smooth = np.mean([np.linalg.norm(self.reduced_data[i,:] - self.reduced_data[j,:]) for (i,j) in zip(linear_ordering[:-1], linear_ordering[1:])])
        self.circ_smooth = np.sum([np.linalg.norm(self.reduced_data[i,:] - self.reduced_data[j,:]) for (i,j) in zip(circ_ordering[:-1], circ_ordering[1:])]) + \
                            np.linalg.norm(self.reduced_data[circ_ordering[0],:] - self.reduced_data[circ_ordering[-1],:])
        self.circ_smooth = self.circ_smooth/self.reduced_data.shape[0]
        self.met_smooth = self.circ_smooth/self.linear_smooth
        
    def plot_true_vs_predicted_phase(self, true_phase, data=None):
        """
        Outputs a plot of CYCLOPS phase against sample collection times
        """
        if data is None:
            data = self.reduced_data
        phase = self.predict_phase(data)
        plt.figure(figsize=(5,4))
        plt.scatter(true_phase, phase, s=8)
        plt.title('Plot of CYCLOPS phase vs sample collection times', size=14)
        plt.xlabel('Sample collection (CT)', size=11)
        plt.ylabel('CYCLOPS Phase (rad)', size=11)
        #plt.xlim([0, 2*np.pi])
        plt.show()


class circular_node(keras.layers.Layer):
    """
    Class implementation of the circular bottleneck layer. This layer has no trainable weights. Class implemented instead of a lambda function.
    """
    def __init__(self):
        super().__init__()
        self._name = 'embedding'
    
    def call(self, inputs):
        return tf.linalg.normalize(inputs, ord=2, axis=1)[0]