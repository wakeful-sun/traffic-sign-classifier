import pickle
import csv
from sklearn.utils import shuffle
import os


class TrafficSignsData:

    def __init__(self, data_folder_path="../traffic-signs-data/"):
        training_file = os.path.join(data_folder_path, "train.p")
        validation_file = os.path.join(data_folder_path, "valid.p")
        testing_file = os.path.join(data_folder_path, "test.p")
        sign_names_file = os.path.join(data_folder_path, "signnames.csv")

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        with open(sign_names_file) as f:
            self.s_names_dict = {}
            read_csv = csv.reader(f, delimiter=",")
            for index, row in enumerate(read_csv):
                if index > 0:
                    self.s_names_dict[int(row[0])] = row[1]

        self.train_data_set = DataSet(train['labels'], train['features'])
        self.valid_data_set = DataSet(valid['labels'], valid['features'])
        self.test_data_set = DataSet(test['labels'], test['features'])

    def print_info(self):
        print("Number of training examples =", self.train.length)
        print("Number of testing examples =", self.test.length)
        print("Number of testing examples =", self.validation.length)
        print("Number of classes =", len(set(self.train.labels)))
        print("Image data shape = {0[1]}x{0[2]}x{0[3]}".format(self.train.features.shape))

    @property
    def sign_names(self):
        return self.s_names_dict

    @property
    def train(self):
        return self.train_data_set

    @property
    def validation(self):
        return self.valid_data_set

    @property
    def test(self):
        return self.test_data_set


class DataSet:

    def __init__(self, labels, features):
        n_l, n_f = len(labels), len(features)
        if n_l != n_f:
            raise Exception("Amount of labels [{}] must be equal to amount of features [{}]".format(n_l, n_f))

        self._f = features
        self._l = labels
        self.offset = 0
        self.current_batch = [], []

    @property
    def length(self):
        return len(self._l)

    @property
    def features(self):
        return self._f

    @property
    def labels(self):
        return self._l

    @property
    def current(self):
        return self.current_batch

    def shuffle(self):
        self._l, self._f = shuffle(self._l, self._f)
        self.reset()

    def apply_map(self, map_func):
        self._l, self._f = map_func(self._l, self._f)
        self.reset()

    def move_next(self, batch_size):
        self.current_batch = self._l[self.offset:self.offset+batch_size], self._f[self.offset:self.offset+batch_size]
        if len(self.current_batch[0]) > 0:
            self.offset = self.offset + batch_size
            return True
        else:
            self.reset()
            return False

    def reset(self):
        self.current_batch = [], []
        self.offset = 0