import h5py

f = h5py.File('./datasets/train_catvnoncat.h5','r')

print(list(f.keys()))
print(f['train_set_y'][:].reshape(1,f['train_set_y'][:].shape[0]))