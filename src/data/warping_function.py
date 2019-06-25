import pandas as pd
import numpy as np
import cv2
from skimage import transform as trans


def warping(img, landmark):
    '''
    Return warped img. Size 112x112

    :param np.array img: Full frame image
    :param np.array landmark: array with 5 key points coordinates of the face

    :return: warped image
    :rtype: np.array
    '''

    image_size = [112, 112]

    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    if image_size[1] == 112:
        src[:, 0] += 8.0

    dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    assert len(image_size) == 2

    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

    return warped


def example():
    '''
    Example of using warping function
    '''

    train_df = pd.read_csv('/path/to/train_df.csv')
    row = train_df.iloc[0]

    img = cv2.imread(row.crop_path)[..., ::-1]

    landmarks5 = np.zeros((5, 2))
    for i, (x, y) in enumerate(zip(['x0', 'x1', 'x2', 'x3', 'x4'],
                                   ['y0', 'y1', 'y2', 'y3', 'y4'])):
        landmarks5[i, 0] = int(row.bbox_x + row[x])
        landmarks5[i, 1] = int(row.bbox_y + row[y])

    warped_img = warping(img, landmarks5)
    cv2.imwrite('/path/to/save/warped_img.jpg', warped_img)
