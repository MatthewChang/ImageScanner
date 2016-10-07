import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pdb
import sys
from sets import Set

PLOT=False

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

#Computes the angle between the vectors a and b traced from a counter clockwise
def angle(a,b):
    a = normalized(a)
    b = normalized(b)
    v = min(np.vdot(a,b),1)
    if(v >= 0):
        return math.acos(v)
    else:
        return math.pi*2 - math.acos(v)

def box(width,height):
    return [np.array([0,height]),np.array([width,height]),np.array([width,0]),np.array([0,0])]

# def growRegion(im,point):
#     F = [point]
#     out_im = np.zeros(im.shape)
#     closed = Set()
#     def point_match(np):
#         return np[0] >= 0 and np[0] < im.shape[0] \
#             and np[1] >= 0 and np[1] < im.shape[1] \
#             and im[np[0],np[1]] <= 50
#     while len(F) > 0:
#         p = F.pop()
#         closed.add(p)
#         out_im[p[0],p[1]] = 1
#         new_points = [(p[0] + x[0],p[1] + x[1]) for x in [(-1,0),(1,0),(0,1),(0,-1)]]
#         for new_point in new_points:
#             if new_point not in closed and point_match(new_point):
#                 F.append(new_point)
#     return out_im

#TODO: Try high passed histrogram looking for trough after first peak, with threshold for what counts as trough as peaks
img = cv2.imread('./test4.jpg')
orig = img.copy()
scale_ratio = 1
if img.shape[0]*img.shape[1] > 250000:
    scale_ratio = 500.0/img.shape[0]
    img = cv2.resize(img, (int(img.shape[1]*scale_ratio),int(img.shape[0]*scale_ratio)))
size = img.shape
print size
#Convert to lab space
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

#Medial blur proportional to image size
blur_size = lab.shape[0]/50
blur_size += 1 if (blur_size % 2 == 0) else 0
blurred_lab = cv2.medianBlur(lab, blur_size)

#Compute distances to center value
key_value = blurred_lab[size[0]/2,size[1]/2,:]
shifted = blurred_lab.astype(np.float32) - key_value.reshape((1,1,3))
for i in range(0,3):
    shifted[:,:,i] = np.multiply(shifted[:,:,i],shifted[:,:,i])
diff = np.sqrt(np.sum(shifted,2)).astype(np.uint8)

#Compute histogram of differences
histr = cv2.calcHist([diff],[0],None,[256],[0,256])

#Low pass filter the histogram only keeping values which are un scaled from the convolution
# i.e. each value from the contributing to the convolved value is valid in the original
window_width = 12
conv_size = 2*window_width + 1
histr_c = np.convolve(histr.transpose()[0],np.ones(conv_size)/conv_size,'valid')

#Find a peak and a trough after that peak, which is significantly lower than the peak
# Perhaps it would be more stable to simply low pass more and ignore the depth of the trough
trough = -1
peak = -1
if(histr_c[1] <= histr_c[0]):
    peak = 0
for i in range(1,len(histr_c)-1):
    if histr_c[i-1] < histr_c[i] and histr_c[i+1] <= histr_c[i]:
        peak = i
    if peak >= 0 and histr_c[i-1] >= histr_c[i] and histr_c[i+1] > histr_c[i] and histr_c[i] < histr_c[peak]/2:
        trough = i
        break

#Threshold based on this trough, note the shift from the window size
#The 0th element in the filter histogram is the average of the first 2*window_width + 1 elements
#In the original histogram
regions = cv2.threshold(diff,trough+window_width+1,255,cv2.THRESH_BINARY_INV)

if PLOT:
    plt.plot(histr_c)
    plt.xlim([0,256])
    plt.axvline(x=trough)
    plt.axvline(x=peak)
    # plt.axhline(y=med)
    plt.show()
    plt.subplot(151),plt.imshow(thumb,cmap = 'gray')
    plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(152),plt.imshow(regions[1],cmap = 'gray')
    plt.title('Thresh Image'), plt.xticks([]), plt.yticks([])

#Find contours in the thresholded image
(cnts, _) = cv2.findContours(regions[1].copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
# loop over the contours
found = False
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        found = True
        break

if(not found):
    print "no document found"
    plt.show()
    sys.exit()

#Order the 4 points from top left clockwise around
centroid = reduce(lambda a,x: a + x,screenCnt)/4
shifted = [x-centroid for x in screenCnt]
top_left = max(shifted, key=lambda x: np.vdot(x,[-1,1]))
ordered = sorted(shifted, key = lambda x: angle(x,top_left))
ordered = [x + centroid for x in ordered]
reduced = [x[0] for x in ordered]
src_pts = np.array(reduced).astype(np.float32)

#scale and start working on original image again
src_pts = src_pts / scale_ratio
#width of the warped image
width = int((src_pts[1][0] - src_pts[0][0] +src_pts[2][0] - src_pts[3][0])/2)
#height of the warped image
height = int((src_pts[0][1] - src_pts[3][1] +src_pts[1][1] - src_pts[2][1])/2)
#Desired shape of the output
out_pts = np.array(box(width,height)).astype(np.float32)
transform = cv2.getPerspectiveTransform(src_pts,out_pts)

#Perform warp and convert to greyscale
warped = cv2.warpPerspective(orig, transform, (width,height))
warped_gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

#Adaptive threshold to get final image
kernel_size = warped_gray.shape[0]/30
kernel_size += 1 if kernel_size % 2 == 0 else 0
final = cv2.adaptiveThreshold(warped_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,kernel_size,15)

dots = orig.copy()
for p in src_pts:
    cv2.circle(dots,tuple(p),20,(255,0,0),20)

cv2.imwrite('final.png',final)
if PLOT:
    plt.subplot(153),plt.imshow(dots,cmap = 'gray')
    plt.title('Dots Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(154),plt.imshow(warped,cmap = 'gray')
    plt.title('Warped Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(155),plt.imshow(final,cmap = 'gray')
    plt.title('Final Image'), plt.xticks([]), plt.yticks([])
    plt.show()
else:
    plt.imshow(final,cmap = 'gray')
    plt.title('Final Image'), plt.xticks([]), plt.yticks([])
    plt.show()
