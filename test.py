import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import pdb
import sys
from sets import Set

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

def growRegion(im,point):
    F = [point]
    out_im = np.zeros(im.shape)
    closed = Set()
    def point_match(np):
        return np[0] >= 0 and np[0] < im.shape[0] \
            and np[1] >= 0 and np[1] < im.shape[1] \
            and im[np[0],np[1]] <= 50
    while len(F) > 0:
        p = F.pop()
        closed.add(p)
        out_im[p[0],p[1]] = 1
        new_points = [(p[0] + x[0],p[1] + x[1]) for x in [(-1,0),(1,0),(0,1),(0,-1)]]
        for new_point in new_points:
            if new_point not in closed and point_match(new_point):
                F.append(new_point)
    return out_im

img = cv2.imread('./test_real.jpg')
if img.shape[0]*img.shape[1] > 250000:
    ratio = 500.0/img.shape[0]
    img = cv2.resize(img, (int(img.shape[1]*ratio),int(img.shape[0]*ratio)))
size = img.shape
print size
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
blur_size = lab.shape[0]/25
blur_size += 1 if (blur_size % 2 == 0) else 0
blurred_lab = cv2.medianBlur(lab, blur_size)
key_value = blurred_lab[size[0]/2,size[1]/2,:]
shifted = blurred_lab.astype(np.float32) - key_value.reshape((1,1,3))
for i in range(0,3):
    shifted[:,:,i] = np.multiply(shifted[:,:,i],shifted[:,:,i])
diff = np.sqrt(np.sum(shifted,2)).astype(np.uint8)
regions = cv2.threshold(diff,20,255,cv2.THRESH_BINARY_INV)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
thumb = cv2.cvtColor(blurred_lab, cv2.COLOR_LAB2BGR)
blurred = cv2.medianBlur(gray, blur_size)
edges = cv2.Canny(diff,0,30)

grown = growRegion(diff,(size[0]/2,size[1]/2))
#plt.hist(diff.ravel(),256,[0,256]); plt.show()

plt.subplot(151),plt.imshow(thumb,cmap = 'gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(diff,cmap = 'gray')
plt.title('diff Image'), plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow(regions[1],cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(154),plt.imshow(grown,cmap = 'gray')
plt.title('Grown Image'), plt.xticks([]), plt.yticks([])


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
        found = False
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
import pdb; pdb.set_trace()
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
