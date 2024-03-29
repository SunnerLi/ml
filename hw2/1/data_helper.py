import numpy as np
import struct

def load_data():
    train_img_path = './train-images-idx3-ubyte'
    train_label_path = 'train-labels-idx1-ubyte'
    test_img_path = './t10k-images-idx3-ubyte'
    test_label_path = 't10k-labels-idx1-ubyte'
    return (read_imgs(train_img_path), read_label(train_label_path)), (read_imgs(test_img_path), read_label(test_label_path))

def read_imgs(path='./train-images-idx3-ubyte'):
    # Read header
    idx_whole_string = open(path, 'rb').read()
    fmt_header = '>iiii'
    offset = 0
    _, batch, height, width = struct.unpack_from(fmt_header, idx_whole_string, offset=offset)

    # Analysis the contain
    imgs = np.zeros([batch, height * width])
    fmt_img = '>' + str(height * width) + 'B'
    offset += struct.calcsize(fmt_header)
    img_size = struct.calcsize(fmt_img)
    for i in range(batch):
        _img_contain = struct.unpack_from(fmt_img, idx_whole_string, offset=offset)
        imgs[i] = np.asarray(_img_contain)
        offset += struct.calcsize(fmt_img)
    return imgs

def read_label(path='train-labels-idx1-ubyte'):
    # Read header
    idx_whole_string = open(path, 'rb').read()
    fmt_header = '>ii'
    offset = 0
    _, batch = struct.unpack_from(fmt_header, idx_whole_string, offset=offset)

    # Read label contain
    tags = np.zeros([batch])
    fmt_label = '>B'
    offset += struct.calcsize(fmt_header)
    for i in range(batch):
        _label_contain = struct.unpack_from(fmt_label, idx_whole_string, offset=offset)
        tags[i] = np.asarray(_label_contain)
        offset += struct.calcsize(fmt_label)
    return tags
