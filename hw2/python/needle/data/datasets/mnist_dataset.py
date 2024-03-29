from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.transforms = transforms
        images = []
        def read_header(f):
            int_bytes = 4
            # read image
            magic_num_buf = f.read(int_bytes)
            magic_num = struct.unpack('>i', magic_num_buf)[0]
            dim = magic_num & 0x000000ff
            int_buf = f.read(int_bytes * dim)
            pack_str = '>' + 'i' * dim
            dims = struct.unpack(pack_str, int_buf)
            return dims

        with gzip.open(image_filename, 'rb') as f:
            dims = read_header(f)
            for i in range(dims[0]):
                length = dims[1] * dims[2]
                buf = f.read(length)
                pack_str = 'B' * length
                r = list(struct.unpack(pack_str, buf))
                images.append(r)

        with gzip.open(label_filename, 'rb') as f:
            dims = read_header(f)
            length = dims[0]
            buf = f.read(length)
            pack_str = 'B' * length
            label = list(struct.unpack(pack_str, buf))
        self.images, self.labels = np.array(images, dtype=np.float32) / 255, np.array(label, dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        r = self.images[index]
        if self.transforms is not None:
            r = r.reshape((28, 28, -1))
            r = self.apply_transforms(r)
            r = r.reshape(-1, 28 * 28)
        return r, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return self.images.shape[0]
        ### END YOUR SOLUTION