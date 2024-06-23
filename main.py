# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import torch
import torchvision
import random

ITERATIONS = 70


# -------------------------Q1-----------------------------
# 1.a
def max_lloyd_quantizer(data, levels, meps):
    """
    The function implements the iterative Max-Llyod algorithm for
    image quantization, and return the quantized image and some
    parameters of the quantization.
    inputs:
    data: one channel image in a uint8 format.
    levels: number of wanted different representation levels.
    meps: minimal required approximation.
    outpus:
    dataout: the image after the quantization.
    distortion: a vector with the size 1 X number of
    iterations. The vector contains the
    average distortion of the quantized
    image in each iteration.
    QL: a vector with the length of levels that contains
    the different representation levels.
    """
    # ====== YOUR CODE: ======
    lower_bound = np.min(data)
    upper_bound = np.max(data)
    r_k = np.sort(np.random.choice(np.ravel(data),levels+1,replace=False))
    # r_k = np.sort(np.random.uniform(lower_bound, upper_bound, levels))
    f_k = [0] * levels
    hist, bins = np.histogram(data.ravel(), bins=256)
    pdf = hist / data.size
    distortion = []
    dataout = np.zeros_like(data)
    counter = 0
    while counter < ITERATIONS:
        for k in range(1,levels+1):
            numenator = sum([x * pdf[x] for x in range(int(r_k[k-1]), int(r_k[k]))])  # mone
            denumenator = sum([pdf[x] for x in range(int(r_k[k-1]), int(r_k[k]))])  # mechane
            if denumenator != 0:
                f_k[k-1] = numenator / denumenator
        for k in range(levels-1):
            r_k[k] = (f_k[k] + f_k[k + 1]) / 2
        distortion.append(np.mean((data - dataout) ** 2))
        for k in range(levels):
            dataout[(r_k[k] <= data) & (data < r_k[k+1])] = f_k[k]
        dataout[(r_k[levels] == data)] = f_k[levels - 1]
        if counter > 0:
            epsilon = np.abs(distortion[counter] - distortion[counter - 1]) / distortion[counter-1]
            if epsilon < meps:
                break
        counter += 1

    QL = f_k

    # ========================
    return dataout, distortion, QL


# 1.a
colorful_img = cv2.imread("../given_data/colorful.jpg")
colorful_rgb = cv2.cvtColor(colorful_img, cv2.COLOR_BGR2RGB)
plt.imshow(colorful_rgb)
plt.show()
colorful_float = colorful_rgb.astype(float)

#1.b
r_dataout, r_distortion, r_QL= max_lloyd_quantizer(colorful_float[:,:,0], 6, 0.01)
g_dataout, g_distortion, g_QL= max_lloyd_quantizer(colorful_float[:,:,1], 6, 0.01)
b_dataout, b_distortion, b_QL= max_lloyd_quantizer(colorful_float[:,:,2], 6, 0.01)
stack = np.dstack((r_dataout, g_dataout, b_dataout)).astype(np.uint8)
rgb = cv2.cvtColor(stack, cv2.COLOR_BGR2RGB)

plt.imshow(r_dataout)
plt.title("6 levels - red channel")
plt.show()
plt.imshow(g_dataout)
plt.title("6 levels - green channel")
plt.show()
plt.imshow(b_dataout)
plt.title("6 levels - blue channel")
plt.show()
plt.imshow(rgb)
plt.title("6 levels - rgb channels")
plt.show()


figure = plt.figure()
our_plot = figure.add_subplot (1,1,1)
our_plot.plot(r_distortion, 'r', label='Red channel')
our_plot.plot(g_distortion, 'g', label='Green channel')
our_plot.plot(b_distortion, 'b', label='Blue channel')
our_plot.legend()
plt.title("6 levels Distortion")
plt.xlabel("iterations")
plt.ylabel("distortion ")
plt.show()

'''
r_dataout, r_distortion, r_QL= max_lloyd_quantizer(colorful_float[:,:,0], 15, 0.01)
g_dataout, g_distortion, g_QL= max_lloyd_quantizer(colorful_float[:,:,1], 15, 0.01)
b_dataout, b_distortion, b_QL= max_lloyd_quantizer(colorful_float[:,:,2], 15, 0.01)
satck = np.dstack((r_dataout, g_dataout, b_dataout)).astype(np.uint8)
rgb = cv2.cvtColor(satck, cv2.COLOR_BGR2RGB)
plt.imshow(r_dataout)
plt.title("15 levels - red channel")
plt.show()
plt.imshow(g_dataout)
plt.title("15 levels - green channel")
plt.show()
plt.imshow(b_dataout)
plt.title("15 levels - blue channel")
plt.show()
plt.imshow(rgb)
plt.title("15 levels - rgb channels")
plt.show()

figure = plt.figure()
our_plot = figure.add_subplot (1,1,1)
our_plot.plot(r_distortion, 'r', label='Red channel')
our_plot.plot(g_distortion, 'g', label='Green channel')
our_plot.plot(b_distortion, 'b', label='Blue channel')
our_plot.legend()
plt.title("15 levels Distortion")
plt.xlabel("iterations")
plt.ylabel("distortion ")
plt.show()


# ------------------------Q2-----------------------------
# 2.a
loaded_images = glob.glob('../given_data/LFW/*.pgm')
images = []
for image in loaded_images:
    img = cv2.imread(image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(gray_img)

# display 4 images:
img1, img2, img3, img4 = images[2010], images[2907], images[7777], images[2204]

plt.imshow(img1, cmap='gray')
plt.title("chosen image with index 2010")
plt.show()

plt.imshow(img2, cmap='gray')
plt.title("chosen image with index 2907")
plt.show()

plt.imshow(img3, cmap='gray')
plt.title("chosen image with index 7777")
plt.show()

plt.imshow(img4, cmap='gray')
plt.title("chosen image with index 2204")
plt.show()

X = [np.reshape(img, img.size, 'F') for img in images]
X = np.stack(X, axis=1)
X_mean = np.mean(X, axis=1)
mean_img = np.reshape(X_mean, images[0].shape, 'F')

plt.imshow(mean_img, cmap='gray')
plt.title("mean img of all images in array")
plt.show()

mu = np.reshape(mean_img, (4096, 1), 'F')
Y = X - mu
Y_cov_matrix = np.cov(Y)


# 2.b

def calc_with_different_k(k, Y_cov_matrix, Y):
    eigen_values, eigen_vectors = np.linalg.eigh(Y_cov_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    eig_vals = eigen_values[sorted_indices[:k]]
    eig_vecs = eigen_vectors[:, sorted_indices[:k]]

    plt.figure()
    plt.plot(list(range(k)), eig_vals)
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title('k largest eigenvalues')
    plt.show()

    eig_vec1 = np.reshape(eig_vecs[:, 0], (64, 64), 'F')
    eig_vec2 = np.reshape(eig_vecs[:, 1], (64, 64), 'F')
    eig_vec3 = np.reshape(eig_vecs[:, 2], (64, 64), 'F')
    eig_vec4 = np.reshape(eig_vecs[:, 3], (64, 64), 'F')

    plt.imshow(eig_vec1, cmap='gray')
    plt.title('eig_vec1')
    plt.show()

    plt.imshow(eig_vec2, cmap='gray')
    plt.title('eig_vec2')
    plt.show()

    plt.imshow(eig_vec3, cmap='gray')
    plt.title('eig_vec3')
    plt.show()

    plt.imshow(eig_vec4, cmap='gray')
    plt.title('eig_vec4')
    plt.show()

    # 2.c
    P = np.transpose(eig_vecs) @ Y

    # 2.d
    x1 = np.reshape(eig_vecs @ P[:, 2010] + np.squeeze(mu), (64, 64), 'F')
    x2 = np.reshape(eig_vecs @ P[:, 2907] + np.squeeze(mu), (64, 64), 'F')
    x3 = np.reshape(eig_vecs @ P[:, 7777] + np.squeeze(mu), (64, 64), 'F')
    x4 = np.reshape(eig_vecs @ P[:, 2204] + np.squeeze(mu), (64, 64), 'F')

    mse_x1 = int(np.sum((images[2010] - x1) ** 2) / images[2010].size)
    mse_x2 = int(np.sum((images[2907] - x1) ** 2) / images[2907].size)
    mse_x3 = int(np.sum((images[7777] - x1) ** 2) / images[7777].size)
    mse_x4 = int(np.sum((images[2204] - x1) ** 2) / images[2204].size)

    plt.imshow(x1, cmap='gray')
    plt.title(f'mse for x1 = {mse_x1}')
    plt.show()

    plt.imshow(x2, cmap='gray')
    plt.title(f'mse for x2 = {mse_x2}')
    plt.show()

    plt.imshow(x3, cmap='gray')
    plt.title(f'mse for x3 = {mse_x3}')
    plt.show()

    plt.imshow(x4, cmap='gray')
    plt.title(f'mse for x4 = {mse_x4}')
    plt.show()


# for k=10:
calc_with_different_k(10, Y_cov_matrix, Y)

# for k=570:
calc_with_different_k(570, Y_cov_matrix, Y)
'''