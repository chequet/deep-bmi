import numpy as np
import torch
import math

class MyIterableDataset(torch.utils.data.IterableDataset):
    """ Implementation of generating data from hail data structures"""

    def __init__(self, bm, ph, batch_list, shuffle, CNN=False):
        self.batch_list = batch_list
        self.bm = bm
        self.shuffle = shuffle
        self.ph = ph
        self.CNN = CNN
        self.on_epoch_end()
        self.start = 0 # default to starting at the beginning
        self.end = len(batch_list) # and ending at the end
        self.counter = 0
        #self.lock = threading.Lock()   #Set self.lock

    def __len__(self):
        # batch_list tells us how many batches in entire set
        return len(self.batch_list)

    def __getitem__(self, index):
        if index >= self.end:
            raise StopIteration
            #with self.lock:
            # get group of columns corresponding to batch at specified index
        print("getting batch at index %i"%index)
        batch = self.batch_list[index]
        X, Y = self.__get_data(batch)
        return X, Y

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.batch_list)

    def __get_data(self, batch):
        X = self.bm.filter_cols(batch).to_numpy()
        #X = X.transpose().astype(int)
        start = batch[0]
        end = batch[-1]+1
        Y = self.ph[start:end]
        return X, Y

    def __iter__(self):
        if self.counter == self.end:
            raise StopIteration
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
            # batch sublist is just whole list
            batch_sublist = self.batch_list
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
            print("worker id %i beginning at batch index %i"%(worker_id,iter_start))
            batch_sublist = self.batch_list[iter_start:iter_end]
        for batch in batch_sublist:
            yield self.__get_data(batch)
            self.counter += 1