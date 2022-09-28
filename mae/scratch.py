import scipy.io as sio
a_dict = {'field1': 0.5, 'field2': 'a string'}
sio.savemat('saved_struct.mat', {'a_dict': a_dict})