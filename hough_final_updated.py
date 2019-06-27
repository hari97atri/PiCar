import RPi.GPIO as GPIO
from time import sleep
import cv2
import picamera
import numpy as np

# configure pins
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(17,GPIO.OUT)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(22,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)

# for camera interfacing
camera = picamera.PiCamera()
camera.resolution = (420, 368)
camera.brightness = 60
camera.start_preview()

i=0  # counter variable

# movement types
def right(sleep_time=0.2):
    GPIO.output(17,GPIO.HIGH)
    GPIO.output(18,GPIO.LOW)
    GPIO.output(22,GPIO.LOW)
    GPIO.output(23,GPIO.LOW)
    sleep(sleep_time)
    stop()

def stop():
    GPIO.output(17,GPIO.LOW)
    GPIO.output(18,GPIO.LOW)
    GPIO.output(22,GPIO.LOW)
    GPIO.output(23,GPIO.LOW)
    return

def left(sleep_time=0.2):
    GPIO.output(17,GPIO.LOW)
    GPIO.output(18,GPIO.HIGH)
    GPIO.output(22,GPIO.LOW)
    GPIO.output(23,GPIO.LOW)
    sleep(sleep_time)
    stop()

def front():
    GPIO.output(17,GPIO.HIGH)
    GPIO.output(18,GPIO.HIGH)
    GPIO.output(22,GPIO.LOW)
    GPIO.output(23,GPIO.LOW)
    return

# take pictures from camera
def take_pictures(i):
    img_sav_name = '/home/pi/PiCar-2019/train_set/Original/' + 'origin_' + str(i) + '.jpeg'
    img = camera.capture(img_sav_name)
    #cv2.imwrite(img_sav_name,img)
    camera.stop_preview()
    return img_sav_name


def draw_lines(img, houghLines, color=[0, 255, 0], thickness=2):                #to draw lines when houghlinesP is used 
        for line in houghLines:                                                 # here it is not used
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),color,thickness)   
                
def draw_lines_new(img, houghLines, color=[0, 255, 0], thickness=2):            #to draw lines when houghlines is used 
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
                    
                
def weighted_img(img, initial_img, α=0.7, β=1.0, λ=0.0):
    return cv2.addWeighted(initial_img, α, img, β, λ)    
    
'''
ang=[]
def angles(houghLines):                                         #print angles when houghlinesP is used  
    for line in houghLines:                                     # here it is not used
        for x1,y1,x2,y2 in line:
            try:
                m=abs((y2-y1)/(x2-x1)+0.0001)
                ang.append(np.tan(m))
            except:
                pass     
        #print(ang)
'''

def angles_left(houghLines):                                    #print angles when houghlines is used (left)
    ang_left=0.0
    for line in houghLines:
        for rho,theta in line:
            #print(theta)
            ang_left=float(90-theta*180/np.pi)
    return(ang_left)
    #print(type(ang_left))
        
    
def angles_right(houghLines):                                   #print angles when houghlinesP is used (right)
    ang_right=0.0
    for line in houghLines:
        for rho,theta in line:
            #print(theta)
            ang_right=float(90-theta*180/np.pi)
    return(ang_right)

        
def right_process(image_right_cropped,id):
    image=image_right_cropped
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/home/pi/PiCar-2019/train_set/grey/grey_right'+ str(id) +'.jpeg',gray_image)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    edges_image = cv2.Canny(blurred_image, 50, 120)
    cv2.imwrite('/home/pi/PiCar-2019/train_set/nd_Image/nd_Image_right'+ str(id) +'.jpeg',edges_image)
    
    rho_resolution = 1
    theta_resolution = np.pi/180
    threshold = 1
    kernel = np.ones((3,3),np.uint8)
    #print("start")
    edges_image = cv2.dilate(edges_image,kernel,iterations = 2)
    cv2.imwrite('/home/pi/PiCar-2019/train_set/edge_Image/edge_Image_right'+ str(id) +'.jpeg',edges_image)
    try:
        hough_lines = cv2.HoughLinesP(edges_image, rho_resolution , theta_resolution , threshold)           #initial edge lines on the original image 
        # print(hough_lines)
        hough_li = hough_lines.tolist()
        #print(type(hough_li))    
    except:
        #print('kali')
        print("empty right half")
        return("right",image_right_cropped,0)
        #hough_li = hough_lines.tolist()
    #cv2.imwrite('/home/pi/PiCar-2019/train_set/Edge_Image/houghp'+ str(id) +'.jpeg',edges_image)
    #print("lenght")
    #hlen=len(hough_li)
    #print(hlen,type(hlen))
    #print(hough_lines)
    hough_lines_image = np.zeros_like(image_right_cropped)
    draw_lines(hough_lines_image, hough_lines)
    original_image_with_hough_lines = weighted_img(hough_lines_image,image)
    final_image = cv2.dilate(original_image_with_hough_lines,kernel,iterations = 2)                 #dilate that to cover the gaps 
    cv2.imwrite('/home/pi/PiCar-2019/train_set/houghp_Image/houghpImage_right'+ str(id) +'.jpeg',final_image)
    
    threshold_n = 30
    try:
        gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
        edges_image = cv2.Canny(blurred_image, 50, 120)
        hough_lines_right = cv2.HoughLines(edges_image, rho_resolution , theta_resolution , threshold_n)
        #print(len(hough_lines_right))
        #use houghline to detect the no. of lines
        leng=len(hough_lines_right)
        lis=hough_lines_right.tolist()
    except Exception as e:
        print(e)
        return("right",image_right_cropped,0)
    #print("\n",hough_lines_right)
    #print(lis)
    rr=0
    aa=0
    neg=0
    pos=0
    ang_lis=[]
    
    for i in lis:
        #print(i[0][0],i[0][1])
        #rr+=i[0][0]
        #aa+=i[0][1]
        ang_lis.append(i[0][1])
    #rr=rr/leng
    #aa=aa/leng
    
    ang_lis_deg=[]
    for i in ang_lis:
        ang_lis_deg.append(float(90-i*180/np.pi))
    #print(ang_lis_deg)
    
    for i in ang_lis_deg:
        if i<0:
            neg+=1
        else:
            pos+=1
    #print(neg,pos)
        
    for i in lis:
        if(neg>pos):
            ag=float(90-(i[0][1])*180/np.pi)
            #print(ag)
            le=neg
            if(ag<0):
                rr+=i[0][0]
                aa+=i[0][1]
                #print(ag)
        elif(pos>neg):
            ag=float(90-(i[0][1])*180/np.pi)
            #print(ag)
            le=pos
            if(ag>0):
                rr+=i[0][0]
                aa+=i[0][1]
    rr=rr/le
    aa=aa/le
    
    #print(rr,aa)
    
    hough_avg_right=np.array([[[rr,aa]]])                                            #take avg to get only line for left image 
    #print(type(hough_avg_left))
    #print(hough_avg_right)
    hough_lines_image_right = np.zeros_like(image_right_cropped)
    #draw_lines_new(hough_lines_image_right, hough_lines_right)
    draw_lines_new(hough_lines_image_right, hough_avg_right)
    original_image_with_hough_lines_right = weighted_img(hough_lines_image_right,image_right_cropped)
    a_right=angles_right(hough_avg_right)                                                                  #to get the angle of the line in left image
    print("a_right = ",a_right)
    return("waste",original_image_with_hough_lines_right,a_right)


def left_process(image_left_cropped,id):
    image=image_left_cropped
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/home/pi/PiCar-2019/train_set/grey/grey_left'+ str(id) +'.jpeg',gray_image)
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    edges_image = cv2.Canny(blurred_image, 50, 120)
    cv2.imwrite('/home/pi/PiCar-2019/train_set/nd_Image/nd_Image_left'+ str(id) +'.jpeg',edges_image)
    
    rho_resolution = 1
    theta_resolution = np.pi/180
    threshold = 1
    kernel = np.ones((3,3),np.uint8)
    
    edges_image = cv2.dilate(edges_image,kernel,iterations = 2)
    cv2.imwrite('/home/pi/PiCar-2019/train_set/edge_Image/edge_Image_left'+ str(id) +'.jpeg',edges_image)
    
    
    try:
        hough_lines = cv2.HoughLinesP(edges_image, rho_resolution , theta_resolution , threshold)           #initial edge lines on the original image 
        # print(hough_lines)
        hough_li = hough_lines.tolist()
        #print(type(hough_li))    
    except:
        #print('kali')
        print("empty left half")
        return("left",image_left_cropped,0)
    
    
    #print(hough_lines)
    hough_lines_image = np.zeros_like(image_left_cropped)
    draw_lines(hough_lines_image, hough_lines)
    original_image_with_hough_lines = weighted_img(hough_lines_image,image_left_cropped)
    final_image = cv2.dilate(original_image_with_hough_lines,kernel,iterations = 2)                 #dilate that to cover the gaps 
    
    cv2.imwrite('/home/pi/PiCar-2019/train_set/houghp_Image/houghpImage_left'+ str(id) +'.jpeg',final_image)
    #cv2.imwrite('/home/pi/PiCar-2019/train_set/Edge_Image/Edge_Image_left'+ str(id) +'.jpeg',final_image)
    
    threshold_n = 30
    
    try:
        #print(final_image)
        gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
        edges_image = cv2.Canny(blurred_image, 50, 120)
        hough_lines_left = cv2.HoughLines(edges_image, rho_resolution , theta_resolution , threshold_n)        #use houghline to detect the no. of lines
        #print(hough_lines_left)
        leng=len(hough_lines_left)
        lis=hough_lines_left.tolist()
    except Exception as e:
        #print(e)
        return("left",image_left_cropped,0)
    
    #print("\n",hough_lines_left)
    #print(lis)
    
    rr=0
    aa=0
    neg=0
    pos=0
    ang_lis=[]
    
    for i in lis:
        #print(i[0][0],i[0][1])
        #rr+=i[0][0]
        #aa+=i[0][1]
        ang_lis.append(i[0][1])
    #rr=rr/leng
    #aa=aa/leng
    
    ang_lis_deg=[]
    for i in ang_lis:
        ang_lis_deg.append(float(90-i*180/np.pi))
    #print(ang_lis_deg)
    
    for i in ang_lis_deg:
        if i<0:
            neg+=1
        else:
            pos+=1
    #print(neg,pos)
        
    for i in lis:
        if(neg>pos):
            ag=float(90-(i[0][1])*180/np.pi)
            #print(ag)
            le=neg
            if(ag<0):
                rr+=i[0][0]
                aa+=i[0][1]
                #print(ag)
        elif(pos>neg):
            ag=float(90-(i[0][1])*180/np.pi)
            #print(ag)
            le=pos
            if(ag>0):
                rr+=i[0][0]
                aa+=i[0][1]
    rr=rr/le
    aa=aa/le
    
    
    hough_avg_left=np.array([[[rr,aa]]])                                            #take avg to get only line for left image 
    #print(type(hough_avg_left))
    #print(hough_avg_left)
    hough_lines_image_left = np.zeros_like(image)
    #draw_lines_new(hough_lines_image_left, hough_lines_left)
    draw_lines_new(hough_lines_image_left, hough_avg_left)
    original_image_with_hough_lines_left = weighted_img(hough_lines_image_left,image)
    a_left=angles_left(hough_avg_left)                                                                  #to get the angle of the line in left image
    print("a_left = ",a_left)
    return("waste",original_image_with_hough_lines_left,a_left)


def image_result(image_path,id):
    try:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        '''
        # uncomment if the webcam is visible
        start_row, start_col = int(0), int(0)
        end_row, end_col = int(height*0.85), int(width)
        cropped_image = image[start_row:end_row, start_col:end_col]
        image = cropped_image
        height, width = image.shape[:2]
        '''
        
        start_row, start_col = int(0.5*height), int(0)
        end_row, end_col = int(height), int(width)
        cropped_image = image[start_row:end_row, start_col:end_col]
        image = cropped_image
        cv2.imwrite('/home/pi/PiCar-2019/train_set/Cropped/Image_'+ str(id) +'.jpeg',image)
        
        height, width = image.shape[:2]
        
        start_row, start_col = int(0), int(0)                                                   #to divide the original into right and left 
        end_row, end_col = int(height), int(width * .5)
        image_left_original = image[start_row:end_row , start_col:end_col]
        
        start_row, start_col = int(0), int(width * .5)
        end_row, end_col = int(height), int(width)
        image_right_original = image[start_row:end_row , start_col:end_col]
        
        label_r,pro_image_r,a_right = right_process(image_right_original,id)
        print("hi1")
        label_l,pro_image_l,a_left = left_process(image_left_original,id)
        print("hi2")

        #print(label_r,label_l,a_right,a_left)
        
        vis = np.concatenate((pro_image_l, pro_image_r),axis=1)
        #print(vis)
        #print("hello")
        cv2.imwrite('/home/pi/PiCar-2019/train_set/Final/Final_Image'+ str(id) +'.jpeg',vis)
        
        if ((label_r == "right")and(label_l == "waste")):
            if (float(a_left) > 0):
                print("right sp")
                front()
                sleep(0.25)
                right()
            if (float(a_left) < 0):
                print("small right")
                #front()
                #sleep(0.25)
                right(0.25)
        elif ((label_l == "left")and(label_r == "waste")):
            if (float(a_right) < 0):
                print("left sp")
                front()
                sleep(0.25)
                left()
            if (float(a_right) > 0):
                print("small left")
                #front()
                #sleep(0.25)
                left(0.25)
        elif ((label_l == "waste")and(label_r == "waste")):
            if(float(abs(a_left)) > 70  and float(abs(a_right)) > 70):
                print("front")
                front()
                sleep(0.25)
                stop()
            elif(float(a_left) > 0 and float(a_right) < 0):                                   #compare the angles and decide 
                print("front")
                front()
                sleep(0.5)
                stop()
            elif(float(a_left) > 0 and float(a_right) > 0):
                print("right")
                front()
                sleep(0.35)
                right()
            elif(float(a_left) < 0 and float(a_right) < 0):
                print("left")
                front()
                sleep(0.35)
                left()
            elif(float(a_right) < 0):
                print("left")
                left()
            elif(float(a_left) > 0):
                print("right")
                right()
            else:
                print("Error")
                stop()
        print("\n")
        return
    except:
        print("None Type Object")
        return

# loop
i=0
while(1):
        i = i + 1
        
        image_sav_name = take_pictures(i)
        #image_sav_name = '/home/pi/img1.jpeg'
        image_result(image_sav_name,i)
            
            
       
