"""
Recognizing the euro coins included in a picture
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import scipy.stats as st


def grayscale(image_array):
    """
    Turning an image to grayscale
    :param image_array: The array containing the original image that we want to turn to grayscale
    :type image_array: np.array
    :return: an image array in grayscale format
    :rtype: np.array
    """
    image_grey = np.zeros((image_array.shape[0], image_array.shape[1])) # Declaring 0's array

    # For loop for grayscaling the image
    for i in range(0, image_grey.shape[0]):
        for j in range(0, image_grey.shape[1]):
            image_grey[i, j] = (image_array[i, j, 0] * 0.25) + (image_array[i, j, 1] * 0.50) + \
                               (image_array[i, j, 2] * 0.25)
    return image_grey


def gauss(image_array):
    """
    Return an image array after applying the gauss blur filter
    :param image_array: An array representing the image that we want to blur out
    :type image_array: np.array
    :return: The image array after applying the gauss blur filter
    :rtype: np.array
    """
    interval = (2 * 3 + 1.) / 10
    x = np.linspace(-3 - interval / 2., 3 + interval / 2., 11)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return ndimage.filters.convolve(image_array, kernel)


def sobel_thinout_doublethreshold(image_array):
    """
    Applying sobel filter, edge thinout and double treshold
    :param image_array: The image array on which we want to apply the filters
    :type image_array: np.array
    :return: An image array after applying the filters
    :rtype: np.array
    """
    # Sobel
    sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    image_vertical = ndimage.filters.convolve(image_array, sobel_vertical)
    image_horizontal = ndimage.filters.convolve(image_array, sobel_horizontal)
    image_array = np.hypot(image_vertical, image_horizontal)
    image_array = (image_array / image_array.max()) * 255 # Sobeled image

    # Thin out
    z = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.int32)
    theta = np.arctan2(image_horizontal, image_vertical) * 180 / np.math.pi
    theta[theta < 0] += 180
    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            if (0 <= theta[i, j] < 22.5) or (157.5 <= theta[i, j] <= 180):
                q = image_array[i, j + 1]
                r = image_array[i, j - 1]
            elif 22.5 <= theta[i, j] < 67.5:
                q = image_array[i + 1, j - 1]
                r = image_array[i - 1, j + 1]
            elif 67.5 <= theta[i, j] < 112.5:
                q = image_array[i + 1, j]
                r = image_array[i - 1, j]
            elif 112.5 <= theta[i, j] < 157.5:
                q = image_array[i - 1, j - 1]
                r = image_array[i + 1, j + 1]

            if image_array[i, j] >= q and image_array[i, j] >= r:
                z[i, j] = image_array[i, j]
            else:
                z[i, j] = 0

    # Double threshold
    high_threshold = z.max() * 0.4
    low_threshold = high_threshold * 0.3

    x, y = np.where(z >= high_threshold)
    i, j = np.where(z < low_threshold)
    k, l = np.where((high_threshold >= z) & (z >= low_threshold))

    image_array[x, y] = 255
    image_array[i, j] = 0
    image_array[k, l] = 25

    return image_array


def hysteresis(image_array):
    """
    Applying hysteresis filter
    :param image_array: An image array on which we want to apply the hysteresis filter
    :type image_array: np.array
    :return: An image array after applying the hysteresis
    :rtype: np.array
    """
    for i in range(1, image_array.shape[0] - 1):
        for j in range(1, image_array.shape[1] - 1):
            if image_array[i, j] == 25:
                if (image_array[i + 1, j - 1] == 255) or (image_array[i + 1, j] == 255) or (
                            image_array[i + 1, j + 1] == 255) or (image_array[i, j - 1] == 255) or (
                            image_array[i, j + 1] == 255) or (image_array[i - 1, j - 1] == 255) or (
                            image_array[i - 1, j] == 255) or (image_array[i - 1, j + 1] == 255):
                    image_array[i, j] = 255
                else:
                    image_array[i, j] = 0
    return image_array


def canny_edge(image_array):
    """
    Applying the full set of methods included in the canny edge.
    These methods are:
        Gauss
        Sobel (Intensity gradient)
        Edge thinout (Non maximum suppresion)
        Double threshold
        Hysterisis (edge tracking)
    :param image_array: An image on which we want to apply the canny edge filter
    :type image_array: np.array
    :return: The image array after applying the canny edge filter
    :rtype: np.array
    """
    return hysteresis((sobel_thinout_doublethreshold(gauss(image_array))))


def euro_2e(image_array, coins, image_draw):
    """
    Finding and drawing 2euro coins
    :param image_array: The image array that we want to search in
    :param coins: A dictionary saving the coins
    :param image_draw: The drawing of the circles on the image
    :return: A tuple containing the coins dictionary as well as the draw image
    """
    # Kernel for voting
    radius = np.zeros((105, 105))
    for i in range(0, 105):
        for j in range(0, 105):
            if 50.5 <= np.math.hypot(i - 51, j - 51) <= 52.5:
                radius[i, j] = 1.0 / 255.0

    radius = ndimage.filters.convolve(image_array, radius)
    radius[radius < 160] = 0

    outline_color = (0, 0, 255)
    cells = []
    for i in range(0, radius.shape[0]):
        for j in range(1, radius.shape[1]):
            if radius[i, j] > 0:
                found = 0
                for k in cells:
                    if np.sqrt(np.power(k[0] - i, 2) + np.power(k[1] - j, 2)) < 13:
                        cells.remove(k)
                        cells.append(((k[0] + i) / 2.0, (k[1] + j) / 2.0))
                        found = 1
                if found == 0:
                    cells.append((i, j))
                    coins['2-Euro'] = coins['2-Euro'] + 1

    for k in cells:
        image_draw.ellipse([(k[1] - 51, k[0] - 51), (k[1] + 51, k[0] + 51)],
            outline=outline_color, width=2)

    return coins, image_draw


def euro_1e(image_array, coins, image_draw):
    """
    Finding and drawing 1euro coins
    :param image_array: The image array that we want to search in
    :param coins: A dictionary saving the coins
    :param image_draw: The drawing of the circles on the image
    :return: A tuple containing the coins dictionary as well as the draw image
    """
    radius = np.zeros((90, 90))
    for i in range(0, 90):
        for j in range(0, 90):
            if 42.85 <= np.math.hypot(i - 43.35, j - 43.35) <= 44.85:
                radius[i, j] = 1.0 / 255.0

    radius = ndimage.filters.convolve(image_array, radius)
    radius[radius < 120] = 0

    outline_color = (0, 255, 0)
    cells = []
    for i in range(0, radius.shape[0]):
        for j in range(1, radius.shape[1]):
            if radius[i, j] > 0:
                found = 0
                for k in cells:
                    if np.sqrt(np.power(k[0] - i, 2) + np.power(k[1] - j, 2)) < 13:
                        cells.remove(k)
                        cells.append(((k[0] + i) / 2.0, (k[1] + j) / 2.0))
                        found = 1
                if found == 0:
                    cells.append((i, j))
                    coins['1-Euro'] = coins['1-Euro'] + 1

    for k in cells:
        image_draw.ellipse([(k[1] - 43, k[0] - 43), (k[1] + 43, k[0] + 43)],
                           outline=outline_color, width=2)

    return coins, image_draw


def euro_50c(image_array, coins, image_draw):
    """
    Finding and drawing 50cent coins
    :param image_array: The image array that we want to search in
    :param coins: A dictionary saving the coins
    :param image_draw: The drawing of the circles on the image
    :return: A tuple containing the coins dictionary as well as the draw image
    """
    radius = np.zeros((97, 97))
    for i in range(0, 97):
        for j in range(0, 97):
            if 46.5 <= np.math.hypot(i - 46.92, j - 46.92) <= 48.5:
                radius[i, j] = 1.0 / 255.0

    radius = ndimage.filters.convolve(image_array, radius)
    radius[radius < 200] = 0

    outline_color = (255, 0, 0)
    cells = []
    for i in range(0, radius.shape[0]):
        for j in range(1, radius.shape[1]):
            if radius[i, j] > 0:
                found = 0
                for k in cells:
                    if np.sqrt(np.power(k[0] - i, 2) + np.power(k[1] - j, 2)) < 13:
                        cells.remove(k)
                        cells.append(((k[0] + i) / 2.0, (k[1] + j) / 2.0))
                        found = 1
                if found == 0:
                    cells.append((i, j))
                    coins['50-Cent'] = coins['50-Cent'] + 1

    for k in cells:
        image_draw.ellipse([(k[1] - 47, k[0] - 47), (k[1] + 47, k[0] + 47)], outline=outline_color, width=2)

    return coins, image_draw


def euro_10c(image_array, coins, image_draw):
    """
    Finding and drawing 10cent coins
    :param image_array: The image array that we want to search in
    :param coins: A dictionary saving the coins
    :param image_draw: The drawing of the circles on the image
    :return: A tuple containing the coins dictionary as well as the draw image
    """
    radius = np.zeros((79, 79))
    for i in range(0, 79):
        for j in range(0, 79):
            if 37.5 <= np.math.hypot(i - 38.25, j - 38.25) <= 39.5:
                radius[i, j] = 1.0 / 255.0

    radius = ndimage.filters.convolve(image_array, radius)
    radius[radius < 140] = 0

    cells = []
    outline_color = (255, 255, 255)
    for i in range(0, radius.shape[0]):
        for j in range(1, radius.shape[1]):
            if radius[i, j] > 0:
                found = 0
                for k in cells:
                    if np.sqrt(np.power(k[0] - i, 2) + np.power(k[1] - j, 2)) < 13:
                        cells.remove(k)
                        cells.append(((k[0] + i) / 2.0, (k[1] + j) / 2.0))
                        found = 1
                if found == 0:
                    cells.append((i, j))
                    coins['10-Cent'] = coins['10-Cent'] + 1

    for k in cells:
        image_draw.ellipse([(k[1] - 38, k[0] - 38), (k[1] + 38, k[0] + 38)], outline=outline_color, width=2)

    return coins, image_draw


def euros(image_array, coins, image_draw):
    """
    Running the individual coin methods
    :param image_array: The image array that we want to search in
    :param coins: A dictionary saving the coins
    :param image_draw: The drawing of the circles on the image
    :return: None
    """
    coins, image_draw = euro_2e(image_array, coins, image_draw)
    coins, image_draw = euro_1e(image_array, coins, image_draw)
    coins, image_draw = euro_50c(image_array, coins, image_draw)
    coins, image_draw = euro_10c(image_array, coins, image_draw)
    print_coins(coins)


def print_coins(coins):
    """
    Printing the dictionary values
    :param coins: A dictionary containing the found coins
    :return: None
    """
    print('There are {2-Euro} 2-Euro coins in the image'.format(**coins))
    print('There are {1-Euro} 1-Euro coins in the image'.format(**coins))
    print('There are {50-Cent} 50-Cent coins in the image'.format(**coins))
    print('There are {10-Cent} 10-Cent coins in the image'.format(**coins))


IMAGES = ['images/coins002.tif', 'images/coins003.tif', 'images/coins004.tif',
          'images/coins005.tif', 'images/coins006.tif', 'images/coins008.tif']

for image in IMAGES:
    print("---------------------- ", image, "---------------------- ")
    IMAGE = Image.open(image)  # Opening image
    IMAGE_GRAYSCALE = canny_edge(grayscale(np.array(IMAGE)/255))  # Image to array, turn colors
    # to [0,1] and canny edge
    DRAW = ImageDraw.Draw(IMAGE) # Drawing original image
    COINS = {'2-Euro': 0, '1-Euro': 0, '50-Cent': 0, '10-Cent': 0} # Dictionary to save the found coins
    euros(IMAGE_GRAYSCALE, COINS, DRAW) # Running the method to find coins and print results
    IMAGE.show() # Showing original image
    print('\n')
