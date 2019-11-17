from algorithm.layer import DeepLearner


class Model(DeepLearner):

    def createModel(self):
        """
        creating the model form the layers
        :return:
        """
        raise NotImplementedError

    def predict(self):
        """
            the method will return the prediction
        :return:
        """
        pass

    def get_evaluation(self):
        """
        evaluation script
        :return:
        """
        pass

