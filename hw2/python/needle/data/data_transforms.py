import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if flip_img: 
            img = np.fliplr(img)
        return img 
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """

        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        print(shift_x, shift_y)
        h, w = img.shape[0], img.shape[1]
        if abs(shift_x) >= h or abs(shift_y) >= w: 
            return np.zeros_like(img)
        img = np.roll(img, -shift_x, axis=0)
        img = np.roll(img, -shift_y, axis=1)
        if shift_x > 0:
            img[-shift_x:, :, :] = 0
        else: 
            img[:-shift_x, :, :] = 0
        if shift_y > 0:
            img[:, -shift_y:, :] = 0
        else: 
            img[:, :-shift_y, :] = 0
        return img
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION
