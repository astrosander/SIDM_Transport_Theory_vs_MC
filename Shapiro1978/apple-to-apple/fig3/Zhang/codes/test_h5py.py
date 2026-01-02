try:
    import h5py
    with h5py.File('test.hdf5', 'w') as f:
        f.create_dataset('test', data=[1, 2, 3])
    open('h5py_works.txt', 'w').write('OK')
except Exception as e:
    open('h5py_error.txt', 'w').write(str(e))

