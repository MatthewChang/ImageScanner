import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pdb
import sys

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

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

img = cv2.imread('./test4.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#blurred = cv2.GaussianBlur(gray, (3, 3), 0)
blurred = cv2.medianBlur(gray, 11)
edges = cv2.Canny(blurred,75,100)
# edges = np.zeros(blurred[:,:,1].shape)
# for i in range(0,3):
#     edges = np.logical_or(edges,cv2.Canny(blurred[:,:,i],75,150) > 100)
# edges = (edges*255).astype(np.uint8)

plt.subplot(151),plt.imshow(blurred,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])


(cnts, _) = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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

centroid = reduce(lambda a,x: a + x,screenCnt)/4
shifted = [x-centroid for x in screenCnt]
top_left = max(shifted, key=lambda x: np.vdot(x,[-1,1]))
ordered = sorted(shifted, key = lambda x: angle(x,top_left))
#shift back
ordered = [x + centroid for x in ordered]
reduced = [x[0] for x in ordered]
src_pts = np.array(reduced).astype(np.float32)
width = int((src_pts[1][0] - src_pts[0][0] +src_pts[2][0] - src_pts[3][0])/2)
height = int((src_pts[0][1] - src_pts[3][1] +src_pts[1][1] - src_pts[2][1])/2)
out_pts = np.array(box(width,height)).astype(np.float32)
transform = cv2.getPerspectiveTransform(src_pts,out_pts)
warped = cv2.warpPerspective(gray, transform, (width,height))
final = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,15)
# show the contour (outline) of the piece of paper
# print "STEP 2: Find contours of paper"
# cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
# cv2.imshow("Outline", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
dots = img.copy()
for p in src_pts:
    cv2.circle(dots,tuple(p),5,(255,0,0))

plt.subplot(153),plt.imshow(dots,cmap = 'gray')
plt.title('Dots Image'), plt.xticks([]), plt.yticks([])
plt.subplot(154),plt.imshow(warped,cmap = 'gray')
plt.title('Warped Image'), plt.xticks([]), plt.yticks([])
plt.subplot(155),plt.imshow(final,cmap = 'gray')
plt.title('Final Image'), plt.xticks([]), plt.yticks([])
plt.show()
