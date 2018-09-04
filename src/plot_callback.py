import keras
from matplotlib import pylab as plt



class PlotCallback(keras.callbacks.Callback):

    def __init__(self, predicition_data):
        super().__init__()
        self.prediction_data = predicition_data
        self.count = 0

    def on_epoch_end(self, epoch, logs={}):
        prediction = self.model.predict(self.prediction_data)
        fig, ax = plt.subplots(1, 1, figsize=(40, 40))
        for idx, im in enumerate(self.prediction_data):
            fig.add_subplot(len(self.prediction_data) * 2, 2, 2 * idx + 1)
            plt.imshow(self.prediction_data[idx], interpolation='nearest')
            fig.add_subplot(len(self.prediction_data) * 2, 2, 2 * idx + 2)
            plt.imshow(prediction[idx].reshape((256, 256)), interpolation='nearest')
        plt.savefig("model/prediction_images/prediction_epoch_{}".format(str(self.count)))
        plt.close(fig)
        self.count +=1


