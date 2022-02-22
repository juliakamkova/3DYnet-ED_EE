import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.misc
import skimage
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import morphology

for Iterat in range(0, 70):

    file_path = 'D:\\Yuliia_data\\Yuliia_data\\lits\\img\\all'
    string = 'volume-' + str(Iterat) + '.nii'
    img_filename = os.path.join(file_path, string)

    img = nib.load(img_filename)
    mymatrix = img.get_fdata()
    size_ct = mymatrix.shape

    print("mymatrix shape = ", size_ct)
    print(type(size_ct[2]))
    # plt.imshow(mymatrix[:,:,20], cmap = "gray")
    # plt.show()

    # matlab code

    # size_ct = size(full_ct);
    # for i = 1:size_ct(1,3)
    #     k(i) = mean(mean(full_ct(:,:,i)));
    # end

    k = []

    for i in range(0, size_ct[2]):
        k.append(np.mean(mymatrix[:, :, i]))

    # print (k)

    # plt.plot(range(0,size_ct[2]),k)
    # plt.show

    result = mymatrix
    result[result < 0] = 0
    result[result > 150] = 0

    # plt.imshow(result[:,:,20], cmap = "gray")
    # plt.show()

    med_denoised = ndimage.median_filter(result, 3)

    k1 = []

    for i in range(0, size_ct[2]):
        k1.append(np.mean(result[:, :, i]))

    k1_ar = np.array(k1)
    # print (k1)
    # plt.plot(range(0,size_ct[2]), k1_ar)
    # plt.show()
    k1_in = k1_ar * (-1)

    pks1, _ = find_peaks(k1_in, distance=10);
    l = []
    l = k1_ar[pks1].astype(int)

    data = dict(zip(l.tolist(), pks1.tolist()))
    LowPeaks = l.tolist()
    LowPeaks.sort()
    LowBorderIndex = LowPeaks[1]
    LowBorder = data[LowBorderIndex]

    plt.plot(range(0, size_ct[2]), k1_ar)
    plt.plot(LowBorder, k1_ar[LowBorder], "x")
    plt.show()
