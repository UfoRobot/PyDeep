import abc


class Dataset:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def next_batch(self, batch_size):
        """
        :return train_images [batch x images_dim], train_labels [batch x labels_dim]
        """
        return


    @abc.abstractmethod
    def get_images(self):
        """
        :return images [batch x images_dim]
        """
        return

    @abc.abstractmethod
    def get_labels(self):
        """
        :return labels [batch x labels_dim]
        """
        return

class MnistTfDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def next_batch(self, batch_size):
        return self.data.next_batch(batch_size)

    def get_images(self):
        return self.data.images

    def get_labels(self):
        return self.data.labels
