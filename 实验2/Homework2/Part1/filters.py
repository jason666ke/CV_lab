import numpy as np
from scipy.signal import correlate2d


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.rot90(kernel, k=2)
    # Compute the height and width of the input image and the kernel
    H, W = image.shape
    k = kernel.shape[0]
    # Pad the edges of the input image with zeros
    pad_size = k // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    # Initialize the output
    conv = np.zeros((H, W))
    # Perform the convolution
    for i in range(H):
        for j in range(W):
            for u in range(k):
                for v in range(k):
                    conv[i, j] += padded_image[i + u, j + v] * kernel[u, v]
    return conv

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    H, W = image.shape
    padded_image = np.zeros((H + 2 * pad_height, W + 2 * pad_width), dtype=image.dtype)
    padded_image[pad_height:-pad_height, pad_width:-pad_width] = image
    return padded_image


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    kernel = np.flip(np.flip(kernel, 1), 0)

    # Get kernel dimensions
    kh, kw = kernel.shape

    # Compute the number of pixels to be padded
    ph = kh // 2
    # pw = kw // 2
    pw = int((kw - 1) / 2)

    # Pad the image with zeros
    padded_image = zero_pad(image, ph, pw)

    # Get the dimensions of the padded image
    ih, iw = padded_image.shape

    # Create an empty output array
    output = np.zeros_like(image)

    # Compute convolution using array operations
    for i in range(ph, ih - ph):
        for j in range(pw, iw - pw):
            output[i - ph, j - pw] = np.sum(padded_image[i - ph:i + ph + 1, j - pw:j + pw + 1] * kernel)

    return output


def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hf, Wf = f.shape
    Hg, Wg = g.shape

    # Flip the kernel g horizontally and vertically
    g_flipped = np.flip(np.flip(g, axis=0), axis=1)

    # Zero-pad the input image f
    pad_height = Hg - 1
    pad_width = Wg - 1
    f_padded = zero_pad(f, pad_height, pad_width)

    # Compute the cross-correlation using convolution
    out = np.zeros((Hf, Wf))
    for i in range(Hf):
        for j in range(Wf):
            # Extract a patch from the input image f_padded
            patch = f_padded[i:i + Hg, j:j + Wg]

            # Compute the dot product of the patch and the kernel g_flipped
            out[i, j] = np.sum(patch * g_flipped)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    # Check if g has odd dimensions, if not, add a row and a column of zeros
    if g.shape[0] % 2 == 0:
        g = np.vstack((g, np.zeros((1, g.shape[1]))))
    if g.shape[1] % 2 == 0:
        g = np.hstack((g, np.zeros((g.shape[0], 1))))

    # Subtract mean of template to make it zero mean
    g = g - np.mean(g)

    # Compute cross-correlation using fast convolution
    corr = conv_fast(f, np.flip(np.flip(g, axis=0), axis=1))

    return corr

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    # Get kernel dimensions
    Hg, Wg = g.shape

    # Compute the number of pixels to be padded
    ph = Hg // 2
    pw = Wg // 2

    # Compute the mean and standard deviation of the template g
    mean_g = np.mean(g)
    std_g = np.std(g)

    # Compute the output shape
    Hf, Wf = f.shape
    Hout = Hf - Hg + 1
    Wout = Wf - Wg + 1

    # Create an empty output array
    out = np.zeros((Hout, Wout))

    # Compute convolution using array operations
    for i in range(Hout):
        for j in range(Wout):
            # Extract patch of f at position (i,j)
            patch_f = f[i:i + Hg, j:j + Wg]

            # Compute the mean and standard deviation of the patch
            mean_patch_f = np.mean(patch_f)
            std_patch_f = np.std(patch_f)

            # Normalize patch_f and g
            patch_f = (patch_f - mean_patch_f) / std_patch_f
            g = (g - mean_g) / std_g

            # Compute the correlation coefficient
            out[i, j] = np.sum(patch_f * g)

    return out
