import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import scipy.misc
from scipy import ndimage
from scipy.signal import find_peaks
from scipy import ndimage
from skimage import exposure

def save_vols(dir, output, i):
    myvol = output[0,:,:,:,0]
    image = nib.Nifti1Image(myvol, np.eye(4))
    nib.save(image, os.path.join(dir, 'vol'+ str(i)+ '.nii.gz'))

file_path = 'D:\\Yuliia_data\\Yuliia_data\\lits\\img\\all'
string = 'volume-0.nii'
img_filename = os.path.join(file_path, string)

img = nib.load (img_filename)
mymatrix = img.get_fdata()
size_ct = mymatrix.shape

print ("mymatrix shape = " , size_ct)
print (type(size_ct[2]))
result = mymatrix
result[result < 0] = 0
result[result > 150] = 0

MeanOrig= []

for i in range(0,size_ct[2]):
    a = result[:,:,i]
    a = a[np.nonzero(a)]
    MeanOrig.append (np.mean(a))

med_denoised = ndimage.median_filter(result, 5)

meanDenoised = []
for i in range(0,size_ct[2]):
    a = med_denoised[:,:,i]
    a = a[np.nonzero(a)]
    meanDenoised.append (np.mean(a))

#save_vols(file_path, med_denoised, 2)

med_denoised = ndimage.median_filter(meanDenoised, 5)
sig_im = exposure.adjust_sigmoid(med_denoised, cutoff=0.1, gain=10, inv=False)
print(sig_im.shape)
meanSegmoid = []

for i in range(0,size_ct[2]):
    a = sig_im [:,:,i]
    a = a[np.nonzero(a)]
    meanSegmoid.append (np.mean(a))

# MeanInvert = np.array(meanDenoised)
# MeanInvert = MeanInvert * (-1)
dist = size_ct[2]/10
pks1, _ = find_peaks(meanSegmoid , distance=dist);
print (pks2)
print(type(pks2))
type(pks2[0])


valuePeaks = []
valuePeaks = np.array(meanSegmoid)[pks1.astype(int)]

print (pks2, valuePeaks)

data = dict(zip (valuePeaks.tolist(), pks_orig.tolist()))
HighPeaks = valuePeaks.tolist()
HighPeaks.sort(reverse = True)
HighBorderIndex = HighPeaks[0]
HighPoint = data[HighBorderIndex]

print (LowPeaks)

plt.plot(range(0,size_ct[2]), meanSegmoid )
plt.plot(HighPoint, meanSegmoid[HighPoint],  "x")
plt.show()


meanSegmoid_inv = np.array(meanSegmoid)

#creting fliped histogram
meanSegmoid_inv = meanSegmoid_inv * (-1)
pks1, _ = find_peaks(meanSegmoid_inv, distance=dist)
print ('piks = ' , pks1)
valuePeaks = []
valuePeaks = meanSegmoid_inv[pks1] #.astype(int)
print ('value of piks = ', valuePeaks)

data = dict(zip (valuePeaks.tolist(), pks1.tolist()))
LowPeaks = valuePeaks.tolist()
LowPeaks.sort(reverse=True)
LowBorderIndex = LowPeaks[0]
i = 1
UpperBorderIndex = LowPeaks[i]

print('sorted value of peaks list = ', LowPeaks)

LowBorder = data[LowBorderIndex]
UpperBorder = data[UpperBorderIndex]

print (LowBorder, UpperBorder)
print(LowPeaks)
mergin = 10

if LowBorder > UpperBorder:
    temp = UpperBorder
    UpperBorder = LowBorder
    LowBorder = temp

while LowBorder > HighPoint:
    i += 1
    LowBorderIndex = LowPeaks[i]
    LowBorder = data[LowBorderIndex]
    UpperBorder = data[UpperBorderIndex]

if (UpperBorder + mergin) <= (size_ct[2] - 1):
    UpperBorder += mergin
else:

    UpperBorder = (size_ct[2] - 1)

plt.plot(range(0, size_ct[2]), meanSegmoid)
plt.plot(LowBorder, meanSegmoid[LowBorder], 'ro')
plt.plot(UpperBorder, meanSegmoid[UpperBorder], 'bo')
plt.show()

string_gt = 'segmentation-0.nii'
img_filename_gt = os.path.join(file_path, string_gt)

img_gt = nib.load (img_filename_gt)
mymatrix_gt = img_gt.get_fdata()
size_ct = mymatrix.shape
# plt.imshow(mymatrix_gt[:,:,60], cmap = "gray")
# plt.show()

mean_gt = []

for i in range(0,size_ct[2]):
    mean_gt.append (np.mean(mymatrix_gt [:,:,i]))

    LB = 0
    UB = 0
    for i in range(0, size_ct[2]):
        if mean_gt[i] != 0 and mean_gt[i - 1] == 0:
            LB = i
        if mean_gt[i] == 0 and mean_gt[i - 1] != 0:
            UB = i
    print(LB, UB)


text2 = ('volume-0.nii' +' find liver from slide '+ str(LowBorder) + ' to '+ str(UpperBorder))


if LB >= LowBorder and UB <= UpperBorder:
    text3 = ('number' + str(Iterat) + ' Liver is inside the bounding box, GOOD!!!')
else:
    text3 = ('number' + str(Iterat) + ' Liver is outside the bounding box, disaster :*(')
print (text3)

with open('Pages.txt', 'w') as f:
    f.write(text1)
    f.write(text2)
    f.write(text3)
f.close()

print (UB, UpperBorder)