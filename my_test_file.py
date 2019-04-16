import cv2
import matplotlib.pyplot as plt
import numpy as np

# return angles of houghlines used
def angles(houghLines):
    ang = 0.0
    for line in houghLines:
        for rho, theta in line:
            ang = float(90-theta*180/np.pi)
    return(ang)

# constants for drawing houghlines
rho_resolution = 1
theta_resolution = np.pi/180
threshold = 55
kernel = np.ones((3,3),np.uint8)

def draw_lines(img, houghLines, color=[0, 255, 0], thickness=2):
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

def image_processing(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    edges_image = cv2.Canny(blurred_image, 50, 120)

    # draw hough lines on the image and capture line parameters
    hough_lines = cv2.HoughLines(edges_image, rho_resolution,
                                        theta_resolution , threshold)

    # reduce multiple hough lines
    length = len(hough_lines)
    lis = hough_lines.tolist()
    xx = 0 
    yy = 0
    for i in lis:
        xx += i[0][0]
        yy += i[0][1]
    xx = xx/length
    yy = yy/length
    hough_avg = np.array([[[xx,yy]]])

    hough_lines_image = np.zeros_like(image) # Blank image
    draw_lines(hough_lines_image, hough_avg) # Draw Hough Lines on blank image

    # Draw Hough Lines on original image
    original_image_with_hough_lines = weighted_img(hough_lines_image, image)
    final_image = cv2.dilate(original_image_with_hough_lines, kernel, iterations=1)

    angle =angles(hough_avg)

    # print the image
    plt.imshow(final_image)
    plt.show()
    print(angle)
    
    return angle

image_path = '/home/hariprasad/IMTR1.jpeg'  # Enter your path here
image = cv2.imread(image_path)

# crop area of interest
height, width = image.shape[:2]
start_row, start_col = int(0.5*height), int(0)
end_row, end_col = int(height), int(width)
cropped_image = image[start_row:end_row , start_col:end_col]

'''
crop the final image to left and right images
'''
# left half
start_row, start_col = int(0), int(0)
end_row, end_col = int(height), int(width * .5) 
cropped_left = cropped_image[start_row:end_row , start_col:end_col]

# right half
start_row, start_col = int(0), int(width * .5)
end_row, end_col = int(height), int(width)
cropped_right = cropped_image[start_row:end_row , start_col:end_col]

'''
Image Processing
'''
a_left = image_processing(cropped_left) # process half of image
a_right = image_processing(cropped_right) # process right half of image

'''
Condition based on angles for decisions
'''
if(float(a_left) > 0 and float(a_right) < 0):   
    print('forward')
elif(float(a_left) > 0 and float(a_right) > 0):
    print('right')
elif(float(a_left) < 0 and float(a_right) < 0):
    print('left')
