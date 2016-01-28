import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m


def luv():
    source_img = cv2.imread('img1.png')
    target_img = cv2.imread('img2.png')
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2LUV)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2LUV)

    # make sure the channel value will be float format
    if source_img.dtype != 'float32':
        img_source = cv2.normalize(source_img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    if target_img.dtype != 'float32':
        target_img = cv2.normalize(target_img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)


    # calculate the size of image
    x_s, y_s, z_s = source_img.shape
    x_t, y_t, z_t = target_img.shape

    # This is all the matrix value that we need
    LMS2LAB = np.matrix(([(1 / m.sqrt(3), 0, 0), (0, 1 / m.sqrt(6), 0), (0, 0, 1 / m.sqrt(2))]))
    ident = np.matrix(([(1, 1, 1,), (1, 1, -2), (1, -1, 0)]))
    ident2 = np.matrix(([(1, 1, 1), (1, 1, -1), (1, -2, 0)]))

    LAB2LMS_final = ident2 * LMS2LAB

    # calculate the mean and std
    mean_sl = np.matrix(source_img[:, :, 0]).mean()
    mean_sa = np.matrix(source_img[:, :, 1]).mean()
    mean_sb = np.matrix(source_img[:, :, 2]).mean()

    std_sl = np.matrix(source_img[:, :, 0]).std()
    std_sa = np.matrix(source_img[:, :, 1]).std()
    std_sb = np.matrix(source_img[:, :, 2]).std()

    mean_tl = np.matrix(target_img[:, :, 0]).mean()
    mean_ta = np.matrix(target_img[:, :, 1]).mean()
    mean_tb = np.matrix(target_img[:, :, 2]).mean()

    std_tl = np.matrix(target_img[:, :, 0]).std()
    std_ta = np.matrix(target_img[:, :, 1]).std()
    std_tb = np.matrix(target_img[:, :, 2]).std()

    std_l = std_tl / std_sl
    std_a = std_ta / std_sa
    std_b = std_tb / std_sb

    res_lab = np.zeros((x_s, y_s, z_s), 'float32')

    for i in range(x_s):
        for j in range(y_s):
            res_lab[i][j][0] = mean_tl + std_l * (source_img[i][j][0] - mean_sl)
            res_lab[i][j][1] = mean_ta + std_a * (source_img[i][j][1] - mean_sa)
            res_lab[i][j][2] = mean_tb + std_b * (source_img[i][j][2] - mean_sb)

    res_img = cv2.cvtColor(res_lab, cv2.COLOR_XYZ2RGB)

    res_img[:, :, 0] = np.clip(res_lab[:, :, 0], 0, 1)
    res_img[:, :, 1] = np.clip(res_lab[:, :, 1], 0, 1)
    res_img[:, :, 2] = np.clip(res_lab[:, :, 2], 0, 1)

    plt.imshow(res_img)
    plt.show()

luv()