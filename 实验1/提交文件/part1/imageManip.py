import numpy as np
import skimage.io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    # Use skimage io.imread
    out = skimage.io.imread(image_path)

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = image[start_row: start_row + num_rows, start_col: start_col + num_cols, :]

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    rows, cols, channels = image.shape
    for row in range(rows):
        for col in range(cols):
            r, g, b = image[row, col]
            image[row, col] = (0.5 * pow(r, 2), g, b)
    out = image

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    row_scale_factor = input_rows / output_rows
    col_scale_factor = input_cols / output_cols

    for row in range(output_rows):
        for col in range(output_cols):
            input_row = int(row * row_scale_factor)
            input_col = int(col * col_scale_factor)
            output_image[row, col] = input_image[input_row, input_col]

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    x, y = point
    new_point = point.copy()
    new_point[0] = x * np.cos(theta) - y * np.sin(theta)
    new_point[1] = x * np.sin(theta) + y * np.cos(theta)

    return new_point


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    # determine center
    cx = input_rows // 2
    cy = input_cols // 2

    row_lower_bound = 0
    row_upper_bound = output_image.shape[0] - 1
    col_lower_bound = 0
    col_upper_bound = output_image.shape[1] - 1

    for row in range(output_image.shape[0]):
        for col in range(output_image.shape[1]):
            # shift pixel position to center
            old_location = np.array([row - cx, col - cy])
            # perform rotation
            new_location = rotate2d(old_location, theta)
            # shift rotated pixel back to the original position
            new_location[0] += cx
            new_location[1] += cy
            # determine whether current coordinates are valid
            if new_location[0] < row_lower_bound or new_location[0] > row_upper_bound:
                continue
            elif new_location[1] < col_lower_bound or new_location[1] > col_upper_bound:
                continue
            else:
                output_image[row, col] = input_image[new_location[0], new_location[1]]

    # 3. Return the output image
    return output_image
