import numpy as np
from sklearn.decomposition import PCA


class KohonenNeuron:
    weights: np.array([])
    row: int
    col: int
    topology_neighbourhood: float

    def __init__(self, set_weights, set_row, set_col):
        self.weights = set_weights
        self.row = set_row
        self.col = set_col

    def topology_distance(self, i, j):
        return (self.row - i)**2 + (self.col - j)**2


class SOM:
    topology_rows: int
    topology_columns: int

    epoch_number: int

    cooperation_coeff: float
    cooperation_decay: float

    learning_rate: float
    learning_decay: float

    neurons: list[KohonenNeuron]

    def __init__(self,
                 set_topology_rows,
                 set_topology_columns,
                 set_epoch_number,
                 set_cooperation_coeff,
                 set_learning_rate,
                 set_cooperation_decay=0.9,
                 set_learning_decay=0.9):

        self.topology_rows = set_topology_rows
        self.topology_columns = set_topology_columns

        self.epoch_number = set_epoch_number

        self.cooperation_coeff = set_cooperation_coeff
        self.learning_rate = set_learning_rate
        self.cooperation_decay = set_cooperation_decay
        self.learning_decay = set_learning_decay

        self.neurons = []

    def generate_neurons(self, X_criterion: np.array([])):
        pca_analizer = PCA(n_components=2)
        pca_analizer.fit(X_criterion)
        mean_vector = np.average(X_criterion, axis=0)
        component_1 = pca_analizer.components_[0]
        component_2 = pca_analizer.components_[1]
        singulars = pca_analizer.singular_values_

        for i in range(self.topology_rows):
            for j in range(self.topology_columns):
                weights = (mean_vector +
                           ((2 * i * singulars[0]) / self.topology_rows - singulars[0]) * component_1 +
                           ((2 * j * singulars[1]) / self.topology_columns - singulars[1]) * component_2)
                new_neuron = KohonenNeuron(weights, i, j)
                self.neurons.append(new_neuron)
        return

    def competition(self, sample: np.array([])):
        min_distance = 10e10
        winner = self.neurons[0]
        for neuron in self.neurons:
            current_distance = np.linalg.norm(neuron.weights - sample)
            if current_distance < min_distance:
                winner = neuron
                min_distance = current_distance
        return winner

    def cooperation(self, winner: KohonenNeuron):
        for neuron in self.neurons:
            h = neuron.topology_distance(winner.row, winner.col)
            neuron.topology_neighbourhood = np.exp(-h / (2 * self.cooperation_coeff**2))
        return

    def adaptation(self, sample: np.array([])):
        for neuron in self.neurons:
            neuron.weights += self.learning_rate * neuron.topology_neighbourhood * (sample - neuron.weights)
        return

    def decay(self):
        self.cooperation_coeff *= self.cooperation_decay
        self.learning_rate *= self.learning_decay

    def fit(self, X_criterion):
        self.generate_neurons(X_criterion)
        for i in range(self.epoch_number):
            X_current = np.copy(X_criterion)
            np.random.shuffle(X_current)
            while np.shape(X_current)[0] > 0:
                sample = X_current[0]
                X_current = np.delete(X_current, 0, axis=0)
                winner = self.competition(sample)
                self.cooperation(winner)
                self.adaptation(sample)
            self.decay()
        return

    def predict(self, X_test: np.array([])):
        prediction = []
        for i in range(np.shape(X_test)[1]):
            winner = self.competition(X_test[1])
            cluster_number = winner.row * self.topology_columns + winner.col
            prediction.append(cluster_number)
        return prediction

    def get_centers(self):
        centers = np.array([])
        for neuron in self.neurons:
            if len(centers) == 0:
                centers = np.copy(neuron.weights)
            else:
                centers = np.vstack([centers, neuron.weights])
        return centers
