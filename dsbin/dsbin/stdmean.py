import numpy as np



class Nomalizer(object):
    def __init__(self, ndarray):
        self.mean = ndarray.mean()
        self.std = ndarray.std()

    def norm(self, ndarray):
        return (ndarray - self.mean) / self.std

    def denorm(self, normed_ndarray):
        return normed_ndarray * self.std + self.mean

train_pkd = np.load('train_labels_2016.npy')
test_pkd = np.load('test_labels_2016.npy')

std = train_pkd.std()
mean = train_pkd.mean()
print(std)
print(mean)

Norm = Nomalizer(train_pkd)
train_pkd = Norm.norm(train_pkd)
test_pkd = Norm.norm(test_pkd)

