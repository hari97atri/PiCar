# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:27:25 2019

@author: VEW1KOR
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def draw_lines(img, houghLines, color=[0, 255, 0], thickness=2):
    for line in houghLines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)   
            
def draw_lines_new(img, houghLines, color=[0, 255, 0], thickness=2):
    for line in houghLines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)   
                
            
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)    

ang=[]
def angles(houghLines):
    for line in houghLines:
        for x1,y1,x2,y2 in line:
            try:
                m=abs((y2-y1)/(x2-x1)+0.0001)
                ang.append(np.tan(m))
            except:
                pass     
    #print(ang)
    
def angles_left(houghLines):
    for line in houghLines:
        for rho,theta in line:
            #print(theta)
            ang_left=(90-theta*180/np.pi)
    print(ang_left)
    

def angles_right(houghLines):
    for line in houghLines:
        for rho,theta in line:
            #print(theta)
            ang_right=(90-theta*180/np.pi)
    print(ang_right)
    

image = cv2.imread("C:/Users/VEW1KOR/Desktop/autosar+can/image30.jpeg")

height, width = image.shape[:2]

start_row, start_col = int(0), int(0)
end_row, end_col = int(height), int(width * .5)
image_left_original = image[start_row:end_row , start_col:end_col]

start_row, start_col = int(0), int(width * .5)
end_row, end_col = int(height), int(width)
image_right_original = image[start_row:end_row , start_col:end_col]


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
edges_image = cv2.Canny(blurred_image, 50, 120)

rho_resolution = 1
theta_resolution = np.pi/180
threshold = 1
kernel = np.ones((3,3),np.uint8)

hough_lines = cv2.HoughLinesP(edges_image, rho_resolution , theta_resolution , threshold)

hough_lines_image = np.zeros_like(image)
draw_lines(hough_lines_image, hough_lines)
original_image_with_hough_lines = weighted_img(hough_lines_image,image)
final_image = cv2.dilate(original_image_with_hough_lines,kernel,iterations = 1)
angles(hough_lines)

height, width = final_image.shape[:2]

start_row, start_col = int(0), int(0)
end_row, end_col = int(height), int(width * .5)
cropped_left = final_image[start_row:end_row , start_col:end_col]

start_row, start_col = int(0), int(width * .5)
end_row, end_col = int(height), int(width)
cropped_right = final_image[start_row:end_row , start_col:end_col]


rho_resolution = 1
theta_resolution = np.pi/180
threshold_n = 70

#plt.imshow(final_image)
image_left = cropped_left
gray_image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
blurred_image_left = cv2.GaussianBlur(gray_image_left, (9, 9), 0)
edges_image_left = cv2.Canny(blurred_image_left, 50, 120)

hough_lines_left = cv2.HoughLines(edges_image_left, rho_resolution , theta_resolution , threshold_n)
leng=len(hough_lines_left)
#print(hough_lines_left)
lis=hough_lines_left.tolist()
#print(lis)
rr=0
aa=0
for i in lis:
    #print(i[0][0],i[0][1])
    rr+=i[0][0]
    aa+=i[0][1]
rr=rr/leng
aa=aa/leng
hough_avg_left=np.array([[[rr,aa]]])
#print(type(hough_avg_left))
#print(hough_avg_left)
hough_lines_image_left = np.zeros_like(image_left)
draw_lines_new(hough_lines_image_left, hough_avg_left)
original_image_with_hough_lines_left = weighted_img(hough_lines_image_left,image_left_original)
angles_left(hough_avg_left)


image_right = cropped_right
gray_image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
blurred_image_right = cv2.GaussianBlur(gray_image_right, (9, 9), 0)
edges_image_right = cv2.Canny(blurred_image_right, 50, 120)

hough_lines_right = cv2.HoughLines(edges_image_right, rho_resolution , theta_resolution , threshold_n)
leng_r=len(hough_lines_right)
#print(hough_lines_right)
lis=hough_lines_right.tolist()
#print(lis)
rrr=0
aar=0
for i in lis:
    #print(i[0][0],i[0][1])
    rrr+=i[0][0]
    aar+=i[0][1]
rrr=rrr/leng_r
aar=aar/leng_r
#print(aar)
hough_avg_right=np.array([[[rrr,aar]]])
#print(type(hough_avg_right))
#print(hough_avg_right)
hough_lines_image_right = np.zeros_like(image_right)
draw_lines_new(hough_lines_image_right, hough_avg_right)
original_image_with_hough_lines_right = weighted_img(hough_lines_image_right,image_right_original)
angles_right(hough_avg_right)

'''

plt.figure(figsize = (30,20))
plt.subplot(321)
plt.imshow(image)
plt.subplot(322)
plt.imshow(edges_image, cmap='gray')
#cv2.imwrite('C:/Users/VEW1KOR/Desktop/autosar+can/edges_image.jpeg',edges_image)

plt.subplot(323)
plt.imshow(final_image, cmap='gray') 

plt.subplot(324)
plt.imshow(original_image_with_hough_lines_left, cmap='gray') 

plt.subplot(325)
plt.imshow(original_image_with_hough_lines_right, cmap='gray') 
'''
#cv2.imwrite('C:/Users/VEW1KOR/Desktop/autosar+can/final_image.jpeg',final_image)
#cv2.imwrite('C:/Users/VEW1KOR/Desktop/autosar+can/final_hough_image.jpeg',original_image_with_hough_lines_new)