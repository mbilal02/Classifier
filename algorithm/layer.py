from abc import ABC, abstractmethod


class DeepLearner(ABC):
    """
    creating a abstract class to get layers
    the object of this class will go to main to get model and prediction
    """
    def __init__(self, layer, sent_max, batch_size, nb_epochs):

        self.layer = layer
        self.sent_max = sent_max
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs


    def addLayer(self):
        """
        The method will get the layers e.g lstm layer, bilstm layer
        :return:
        """
        raise NotImplementedError
    @abstractmethod
    def createModel(self):
        pass
    @abstractmethod
    def predict(self):
        pass
    @abstractmethod
    def get_evaluation(self):
        pass







