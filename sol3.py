import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
import os

GRAYSCALE = 1
MIN_RES = 16


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Construct a Gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter
    :return: tuple(pyr, filter_vec) where:
             pyr: the  resulting  pyramid as  a  standard  python  array with maximum length of max_levels,
                  where each element of the array is a grayscale image.
             filter_vec: normalized row vector of shape(1, filter_size) used for the pyramid construction
    """
    pyr = [im]
    filter_vec = create_filter(filter_size)
    i = 1  # counter to keep track of levels
    rows, cols = im.shape
    while i < max_levels and min(rows, cols) > MIN_RES:
        pyr.append(reduce(pyr[i - 1], filter_vec))
        rows, cols = pyr[i].shape
        i += 1
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Construct a Laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter
    :return: tuple(pyr, filter_vec) where:
             pyr: the  resulting  pyramid as  a  standard  python  array with maximum length of max_levels,
                  where each element of the array is a grayscale image.
             filter_vec: normalized row vector of shape(1, filter_size) used for the pyramid construction
    """
    pyr = []
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(min(max_levels - 1, len(gauss_pyr) - 1)):
        pyr.append(gauss_pyr[i] - expand(gauss_pyr[i + 1], filter_vec))  # according to formula
    pyr.append(gauss_pyr[-1])
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid generated from build_laplacian_pyramid
    :param filter_vec: normalized row vector generated from build_laplacian_pyramid
    :param coeff: list with same number of elements as in lpyr of coefficients
    :return: The original image
    """
    lpyr = [level * coe for level, coe in zip(lpyr, coeff)]  # multiply each level by coeff
    res = lpyr[-1]
    for i in reversed(range(len(lpyr) - 1)):
        res = expand(res, filter_vec) + lpyr[i]
    return res


def render_pyramid(pyr, levels):
    """
    Renders a big image of horizontally stacked pyramid levels
    :param pyr: Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result <= max_levels
    :return: single black image with pyramid levels stacked horizontally
    """
    pyr[0] = (pyr[0] - pyr[0].min()) / (pyr[0].max() - pyr[0].min())  # stretch values to [0,1]
    images = [pyr[0]]
    num_of_rows = pyr[0].shape[0]  # used to calc num of rows for other levels
    for i in range(1, levels):
        pyr[i] = (pyr[i] - pyr[i].min()) / (pyr[i].max() - pyr[i].min())  # stretch values to [0,1]
        new_level = np.zeros((num_of_rows, pyr[i].shape[1]))  # array of zeros of new size to insert to
        new_level[:pyr[i].shape[0]] = pyr[i]  # insert to array
        images.append(new_level)
    return np.hstack(images)


def display_pyramid(pyr, levels):
    """
    Renders and displays a horizontally stacked pyramid image
    :param pyr: Gaussian or Laplacian pyramid
    :param levels: number of levels to present in the result <= max_levels
    """
    plt.figure()
    plt.imshow(render_pyramid(pyr, levels), cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Pyramid blending to combine two images according to a mask. im1, im2 and mask should have
    the same dimensions and be multiples of 2**(max_levels-1)
    :param im1: first grayscale image to be blended
    :param im2: second grayscale image to be blended
    :param mask: boolean mask representing which parts of im1 and im2 should appear in the result
    :param max_levels: max levels passed to laplacian and gaussian pyramids
    :param filter_size_im: size of gaussian filter used in construction of im1 and im2 Laplacian pyramids
    :param filter_size_mask: size of gaussian filter used in construction of mask Gaussian pyramid
    :return: the resulting blended image
    """
    im1_lap, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_lap = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    mask_gaus = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    l_out = []
    for l1, l2, g_m in zip(im1_lap, im2_lap, mask_gaus):
        l_out.append(g_m * l1 + (1 - g_m) * l2)
    l_out = laplacian_to_image(l_out, filter_vec, [1] * len(l_out))
    return np.clip(l_out, a_min=0, a_max=1)


def blending_example1():
    """
    Creates the first blending example
    :return: im1 and im2 as the used images, mask as the binary mask and the result blended image
    """
    im1 = read_image(relpath('externals/trump.jpg'), 2)
    im2 = read_image(relpath('externals/it_clown.jpg'), 2)
    mask = read_image(relpath('externals/mask_1.jpg'), GRAYSCALE)
    mask = mask.round().astype(np.bool)
    res = blend(im1, im2, mask)
    return im1, im2, mask, res


def blending_example2():
    """
    Creates the second blending example
    :return: im1 and im2 as the used images, mask as the binary mask and the result blended image
    """
    im1 = read_image(relpath('externals/coronabeer.jpg'), 2)
    im2 = read_image(relpath('externals/coronamask.jpg'), 2)
    mask = read_image(relpath('externals/mask_2.jpg'), GRAYSCALE)
    mask = mask.round().astype(np.bool)
    res = blend(im1, im2, mask)
    return im1, im2, mask, res


# ---------- Helper functions ---------------------

def create_filter(filter_size):
    """
    Creates a Gaussian filter of size filter_size
    :param filter_size: the size of the Gaussian filter
    :return: normalized row vector of shape(1, filter_size)
    """
    if filter_size == 1:
        return np.array([[1]])
    base_vec = np.convolve([1, 1], [1, 1])
    filter_vec = base_vec
    levels = (filter_size - 1) // 2  # this is the number of levels for the convolution
    for i in range(levels - 1):
        filter_vec = np.convolve(filter_vec, base_vec)
    filter_vec = filter_vec / 2 ** (2 * levels)  # normalize
    return filter_vec.reshape(1, filter_size)


def reduce(im, filter_vec):
    """
    Blurs and subsample an image to reduce it by half
    :param im: a grayscale image with double values in [0,1]
    :param filter_vec: normalized row vector of shape(1, filter_size) to blur with
    :return: reduced image
    """
    blurred_im = convolve(im, filter_vec)  # conv as row vector
    blurred_im = convolve(blurred_im, filter_vec.T)  # conv as col vector
    return blurred_im[::2, ::2]


def expand(im, filter_vec):
    """
    Zero pads and blurs an image to expand it by 2
    :param im: a grayscale image with double values in [0,1]
    :param filter_vec: row vector of shape(1, filter_size) to blur with
    :return: expanded image
    """
    expanded_im = np.zeros((2 * im.shape[0], 2 * im.shape[1]))  # zeros array of twice the orig size
    expanded_im[::2, ::2] = im  # fill with im at even indices
    expanded_im = convolve(expanded_im, 2 * filter_vec)  # conv as row vector
    return convolve(expanded_im, 2 * filter_vec.T)  # conv as col vector


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should be a
    grayscale image (1) or an RGB image (2)
    :return: the image represented by a matrix of type np.float64
    """
    im = imread(filename)
    if representation == GRAYSCALE and im.ndim == 3:
        im = rgb2gray(im)
        return im
    im = im.astype(np.float64)
    im /= 255
    return im


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blend(im1, im2, mask):
    """
    Blends and shows the given images according to mask
    :param im1: first image
    :param im2: second image
    :param mask: binary mask
    :return: result blend
    """
    res = []
    for i in range(3):
        res.append(pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 7, 5, 5))
    res = np.dstack(res)
    fig, a = plt.subplots(nrows=2, ncols=2)
    a[0][0].imshow(im1, cmap='gray')
    a[0][1].imshow(im2, cmap='gray')
    a[1][0].imshow(mask, cmap='gray')
    a[1][1].imshow(res, cmap='gray')
    plt.show()
    return res


if __name__ == '__main__':
    im = read_image('ex3_presubmit/presubmit_externals/monkey.jpg', GRAYSCALE)
    display_pyramid(build_laplacian_pyramid(im, 5, 5)[0], 5)
