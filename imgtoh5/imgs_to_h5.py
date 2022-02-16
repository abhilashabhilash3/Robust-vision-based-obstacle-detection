import os
import sys
import h5py
import numpy as np

meta = list()

#Loop over each row of file name

with open("/home/abhilash/Documents/data/WZL/chesstraining/files.txt", "r") as a_file:
  for line in a_file:
    meta.append(tuple(line.split())) 
 

#print(meta)
metadata = np.rec.array(meta, dtype=[('times', '<u8')])

meta_filename = os.path.join("/home/abhilash/Documents/data/WZL/chesstraining/", "metadata_cache.h5")

with h5py.File(meta_filename, "w") as hf: 
	hf.create_dataset("times", data=metadata.times)
