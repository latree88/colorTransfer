import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import copy

source_img = cv2.imread('img1.png')
target_img = cv2.imread('img2.png')
# img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
# img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)


RGB2LMS = np.matrix(((0.3811, 0.5783, 0.0402), (0.1967, 0.7244, 0.0782), (0.0241, 0.1288, 0.8444)))
LMS2RGB = np.matrix(((4.4679, -3.5873, 0.1193), (-1.2186, 2.3809, -0.1624), (0.0497, -0.2439, 1.2045)))

LMS2lab1 = np.matrix(((1/math.sqrt(3), 0, 0), (0, 1/math.sqrt(6), 0), (0, 0, 1/math.sqrt(2))))
LMS2lab2 = np.matrix(((1, 1, 1), (1, 1, -2), (1, -1, 0)))
LMS2lab = LMS2lab1 * LMS2lab2

lab2LMS1 = np.matrix(((1, 1, 1), (1, 1, -1), (1, -2, 0)))
lab2LMS2 = np.matrix(((math.sqrt(3)/3, 0, 0), (0, math.sqrt(6)/6, 0), (0, 0, math.sqrt(2)/2)))
lab2LMS = lab2LMS1 * lab2LMS2

width_s, height_s, channel_s = source_img.shape
width_t, height_t, channel_t = target_img.shape


source_R = []
source_G = []
source_B = []

target_R = []
target_G = []
target_B = []

for i in range(width_s):
    temp1_s = []
    temp2_s = []
    temp3_s = []
    for j in range(height_s):
        temp1_s.append(source_img[i][j][0])
        temp2_s.append(source_img[i][j][1])
        temp3_s.append(source_img[i][j][2])

    source_R.append(temp1_s)
    source_G.append(temp2_s)
    source_B.append(temp3_s)


source_X = copy.copy(source_R)
source_Y = copy.copy(source_G)
source_Z = copy.copy(source_B)

for i in range(width_t):
    temp1_t = []
    temp2_t = []
    temp3_t = []

    for j in range(height_t):
        temp1_t.append(target_img[i][j][0])
        temp2_t.append(target_img[i][j][1])
        temp3_t.append(target_img[i][j][2])

    target_R.append(temp1_t)
    target_G.append(temp2_t)
    target_B.append(temp3_t)

target_X = copy.copy(target_R)
target_Y = copy.copy(target_G)
target_Z = copy.copy(target_B)


##############################################################
# a = [[R[0][0]], [G[0][0]], [B[0][0]]]
# b = np.matrix(a)
# print b
# convert source image RGB to lms, then use the log10 to convert it to LMS
for i in range(len(source_R)):
    for j in range(len(source_R[i])):
        temp_s_rgb_to_lms = [[source_R[i][j]], [source_G[i][j]], [source_B[i][j]]]
        temp_s_rgb = np.matrix(temp_s_rgb_to_lms)
        source_lms = RGB2LMS * temp_s_rgb
        source_X[i][j] = float(math.log10(source_lms[0][0]))
        source_Y[i][j] = float(math.log10(source_lms[1][0]))
        source_Z[i][j] = float(math.log10(source_lms[2][0]))


# convert target image RGB to lms, then use the log10 to convert it to LMS
for i in range(len(target_R)):
    for j in range(len(target_R[i])):
        temp_t_rgb_to_lms = [[target_R[i][j]], [target_G[i][j]], [target_B[i][j]]]
        temp_t_rgb = np.matrix(temp_t_rgb_to_lms)
        target_lms = RGB2LMS * temp_t_rgb
        target_X[i][j] = float(math.log10(target_lms[0][0]))
        target_Y[i][j] = float(math.log10(target_lms[1][0]))
        target_Z[i][j] = float(math.log10(target_lms[2][0]))


#################################################################
# convert source image from LMS to lab
for i in range(len(source_R)):
    for j in range(len(source_R[i])):
        temp_s_lms_to_lab = [[source_X[i][j]], [source_Y[i][j]], [source_Z[i][j]]]
        temp_s_lab = np.matrix(temp_s_lms_to_lab)
        source_lab = LMS2lab * temp_s_lab
        source_R[i][j] = float(source_lab[0][0])
        source_G[i][j] = float(source_lab[1][0])
        source_B[i][j] = float(source_lab[2][0])


# convert target image form LMS to lab
for i in range(len(target_R)):
    for j in range(len(target_R[i])):
        temp_t_lms_to_lab = [[target_X[i][j]], [target_Y[i][j]], [target_Z[i][j]]]
        temp_t_lab = np.matrix(temp_t_lms_to_lab)
        target_lab = LMS2lab * temp_t_lab
        target_R[i][j] = float(target_lab[0][0])
        target_G[i][j] = float(target_lab[1][0])
        target_B[i][j] = float(target_lab[2][0])


##################################################################

s_mean_l = np.matrix(source_R).mean()
s_mean_a = np.matrix(source_G).mean()
s_mean_b = np.matrix(source_B).mean()

t_mean_l = np.matrix(target_R).mean()
t_mean_a = np.matrix(target_G).mean()
t_mean_b = np.matrix(target_B).mean()

s_std_l = np.matrix(source_R).std()
s_std_a = np.matrix(source_G).std()
s_std_b = np.matrix(source_B).std()

t_std_l = np.matrix(target_R).std()
t_std_a = np.matrix(target_G).std()
t_std_b = np.matrix(target_B).std()

std_res_R = t_std_l/s_std_l
std_res_G = t_std_a/s_std_a
std_res_B = t_std_b/s_std_b


for i in range(len(source_R)):
    for j in range(len(source_R[i])):
        source_R[i][j] = t_mean_l + std_res_R * (source_R[i][j] - s_mean_l)
        source_G[i][j] = t_mean_a + std_res_G * (source_G[i][j] - s_mean_a)
        source_B[i][j] = t_mean_b + std_res_B * (source_B[i][j] - s_mean_b)


########################################################################
# from lab to LMS
for i in range(len(source_R)):
    for j in range(len(source_R[i])):
        temp_s_lab_to_lms = [[source_R[i][j]], [source_G[i][j]], [source_B[i][j]]]
        temp_s_lms = np.matrix(temp_s_lab_to_lms)
        source_lab_to_lms = lab2LMS * temp_s_lms
        source_X[i][j] = float(source_lab_to_lms[0][0])
        source_Y[i][j] = float(source_lab_to_lms[1][0])
        source_Z[i][j] = float(source_lab_to_lms[2][0])


# convert power of 10
for i in range(len(source_X)):
    for j in range(len(source_X[i])):
        source_X[i][j] = math.pow(10.0, source_X[i][j])
        source_Y[i][j] = math.pow(10.0, source_Y[i][j])
        source_Z[i][j] = math.pow(10.0, source_Z[i][j])



# from lms to RGB
res_R = copy.copy(source_X)
res_G = copy.copy(source_Y)
res_B = copy.copy(source_Z)

for i in range(len(source_X)):
    for j in range(len(source_X[i])):
        temp_s_lms_to_rgb = [[source_X[i][j]], [source_Y[i][j]], [source_Z[i][j]]]
        temp_s_rgb_final = np.matrix(temp_s_lms_to_rgb)
        source_lab_to_lms_final = LMS2RGB * temp_s_rgb_final
        res_R[i][j] = float(source_lab_to_lms_final[0][0])
        res_G[i][j] = float(source_lab_to_lms_final[1][0])
        res_B[i][j] = float(source_lab_to_lms_final[2][0])

# print res_R[125][302]


res_img = copy.copy(source_img)
for i in range(width_s):
    for j in range(height_s):
        res_R[i][j] = np.clip(res_R[i][j], 0, 255)
        res_G[i][j] = np.clip(res_G[i][j], 0, 255)
        res_B[i][j] = np.clip(res_B[i][j], 0, 255)

        res_img[i][j][0] = int(res_R[i][j])
        res_img[i][j][1] = int(res_G[i][j])
        res_img[i][j][2] = int(res_B[i][j])


img3 = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)


plt.imshow(img3)
plt.show()
