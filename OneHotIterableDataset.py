import numpy as np
import torch
import math
import os

class OneHotIterableDataset(torch.utils.data.IterableDataset):
    """ Implementation of generating data from hail data structures"""

    def __init__(self, filepath, shuffle):
        self.filepath = filepath
        self.shuffle = shuffle
        self.files = os.listdir(filepath)
        # shuffle file order if shuffle
        if self.shuffle:
            np.random.shuffle(self.files)
        self.start = 0 # default to starting at the beginning
        self.end = len(self.files) # and ending at the end
        self.counter = 0
        #self.lock = threading.Lock()   #Set self.lock

    def __len__(self):
        # how many batches I will return
        return len(self.files)

    def __one_hot(self, data):
        # convert single array of data to one-hot encoding, flatten
        shape = (len(data), 3) 
        onehot = np.zeros(shape) 
        rows = np.arange(len(data)) 
        onehot[rows, data] = 1 
        return onehot.flatten() 


    def __getitem__(self, index):
        # # could implement check to see if batch list is longer than 
        # # number of files at filepath location
        if index >= self.end:
            return
        # get batch at specified index
        batch = self.files[index]
        #print("getting batch %s"%batch)
        filename = self.filepath + batch
        load = np.load(filename)
        # load and convert to list
        X = load['x'].tolist()
        Y = load['y'].astype('float32')
        # in-place edit batches to one-hot encoding, convert back to array of arrays
        X = np.array([self.__one_hot(batch) for batch in X])
        return X, Y

    def __iter__(self):
        
        # # shuffle if you've reached the end of the epoch
        # if (self.counter == (self.end-1) and self.shuffle):
        #     files = np.random.shuffle(self.files)
        # if multiple workers, split workload
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: 
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
            self.files = self.files[iter_start:iter_end]
            #print("worker id %i handling %i files"%(worker_id,self.__len__()))
        while self.counter < self.__len__():
            #print("worker id %i, file count %i, file %s"%(worker_id,self.counter,(self.filepath + self.files[self.counter])))
            yield self.__getitem__(self.counter)
            self.counter += 1
        # stop if you've returned everything
        #print("ran out of files")
        raise StopIteration