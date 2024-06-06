class EffectEmbeddingDataset(torch.utils.data.IterableDataset):

    def __init__(self, filepath, files, shuffle, method, beta_mask):
        self.filepath = filepath
        self.shuffle = shuffle
        self.files = files
        # shuffle file order if shuffle
        if self.shuffle:
            np.random.shuffle(self.files)
        self.start = 0 # default to starting at the beginning
        self.end = len(self.files) # and ending at the end
        self.counter = 0
        self.method = method
        self.beta_mask = beta_mask
        #print("embedding method %i" %self.method)
        #self.lock = threading.Lock()   #Set self.lock

    def __len__(self):
        # how many batches I will return
        return len(self.files)


    def __effect_embedding(self, data):
        if self.method == 1:
            mag1 = 0.7
            mag2 = 1
        elif self.method == 2:
            mag1 = 1
            mag2 = 2
        # convert single array of data to desired encoding, flatten
        shape = (len(data), 2) 
        embed = np.zeros(shape) 
        # get positive and negative alt effect loci 
        pos = np.where(self.beta_mask==1)
        neg = np.where(self.beta_mask==-1)
        # get loci for each genotype
        homref = np.where(data==0)
        het = np.where(data==1)
        homalt = np.where(data==2)
        # get intersects
        pos_homref = np.intersect1d(pos,homref)
        pos_homalt = np.intersect1d(pos,homalt)
        neg_homref = np.intersect1d(neg,homref)
        neg_homalt = np.intersect1d(neg,homalt)
        # fill in embedding array
        embed[het] = [mag1, mag1]
        embed[pos_homref] = [0, mag2]
        embed[pos_homalt] = [mag2, 0]
        embed[neg_homref] = [mag2, 0]
        embed[neg_homalt] = [0, mag2]
        return embed.flatten()

    def __getitem__(self, index):
        if index >= self.end:
            return
        # get batch at specified index
        batch = self.files[index]
        #print("getting batch %s"%batch)
        filename = self.filepath + batch
        load = np.load(filename)
        # load and convert to list
        X = load['x']
        Y = load['y'].astype('float32')
        # in-place edit batches to one-hot encoding, convert back to array of arrays
        X = np.array([self.__effect_embedding(sample) for sample in X])
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
