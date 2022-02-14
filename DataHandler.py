import pandas as pd
import numpy as np
import random
import math
from Regression import fit_poly

random.seed(11)

class DataHandler:

    data: np.ndarray

    def __init__(self, data_filepath: str):
        # Open data file
        dataframe: pd.DataFrame = pd.read_csv(data_filepath)
        dataframe: pd.DataFrame = dataframe[["close"]]
        # Flip right way around
        self.data = np.flipud(dataframe.to_numpy(np.float32))
    # Retrieve a random sample of n consecutive prices along with target prices
    # n = size of sample
    # n = number of future prices to retrieve after sample
    def sample_random(self, n: int = 192, n_future: int = 1):
        # Get random index and leave room for additional prices
        rand = math.floor(random.random()*(self.data.shape[0] - n - n_future))
        # Get sample and indices
        sample = self.data[rand:rand+n, [0]]
        indices = np.arange(-n/2,n/2).T
        return (np.c_[indices, sample], self.data[rand+n:rand+n+n_future, [0]])

    # Generate a batch of n training samples
    def generate_batch(self, n: int = 1024, sample_n:int = 48, buy_delta: float = 0.02, sell_delta: float = 0.02):
        train_samples = []
        actions = []
        # Keep track of how many actions are stays
        num_stays = 0
        for i in range(n):
            sample, next_price = self.sample_random(n=sample_n+1)
            coefs = fit_poly(sample, 8)
            #print(sample)
            action = np.array([[0], [1], [0]])
            # Get action
            if next_price >= (1 + buy_delta) * sample[sample_n, 1]:
                action = np.array([[1], [0], [0]])
            elif next_price <= (1 - sell_delta) * sample[sample_n, 1]:
                action = np.array([[0], [0], [1]])
            else:
                num_stays += 1
            # Get co-efficients of corresponding polynomial
            sample = np.c_[sample[0:sample_n, 0], np.divide(sample[1:sample_n + 1, 1] - sample[0:sample_n, 1], sample[1:sample_n + 1, 1])]
            train_samples.append(coefs.T)
            actions.append(action)
            # Output progress
            progress = math.ceil(i / n * 100)
            if progress % 5 == 0:
                print('\r', f'Generating batch... {progress}%', end='')
        print(' \n')
        return (np.array(train_samples), np.array(actions), num_stays)

    def generate_batch_diff(self, n: int = 1024, sample_n:int = 48, buy_delta: float = 0.02, sell_delta: float = 0.02):
        train_samples = []
        actions = []
        # Keep track of how many actions are stays
        num_stays = 0
        for i in range(n):
            sample, next_price = self.sample_random(n=sample_n+1)
            #print(sample)
            action = np.array([[0], [1], [0]])
            # Get action
            if next_price >= (1 + buy_delta) * sample[sample_n, 1]:
                action = np.array([[1], [0], [0]])
            elif next_price <= (1 - sell_delta) * sample[sample_n, 1]:
                action = np.array([[0], [0], [1]])
            else:
                num_stays += 1
            # Get co-efficients of corresponding polynomial
            sample = np.c_[sample[0:sample_n, 0], np.divide(sample[1:sample_n + 1, 1] - sample[0:sample_n, 1], sample[1:sample_n + 1, 1])]
            train_samples.append(sample)
            actions.append(action)
            # Output progress
            progress = math.ceil(i / n * 100)
            if progress % 5 == 0:
                print('\r', f'Generating batch... {progress}%', end='')
        print(' \n')
        train_samples = train_samples[:, :, [1]]
        return (np.array(train_samples), np.array(actions), num_stays)
