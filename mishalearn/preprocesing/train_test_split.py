import numpy as np
import pandas as pd


def train_test_split(*arrays: pd.DataFrame | pd.Series, test_size=0.25, random_state=None, shuffle=True):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    test_size = int(n_samples * test_size) if isinstance(test_size, float) else test_size

    splited_arrays = []
    for arr in arrays:
        if arr.shape[0] != n_samples:
            raise ValueError("Все массивы должны иметь одинаковое количество объектов по оси 0.")

        arr = arr.iloc[indices]

        arr_test, arr_train = arr.iloc[:test_size], arr.iloc[test_size:]
        splited_arrays.append(arr_train)
        splited_arrays.append(arr_test)

    return splited_arrays
