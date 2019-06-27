import cv2
import subprocess
import time

threshold_score=0.0001
flag=1
while(1):
    
    if flag != 0:
        start_time = time.time()
        camera = cv2.VideoCapture(0)
        for i in range(10):
            return_value, image = camera.read()
            if i == 9:
                cv2.imwrite('webcam'+str(i)+'.png', image)
        print("image ready\n")
        del(camera)
    try:
        k=open('width_height.txt','r')
        width_height=k.read()
        if width_height:
            print('the width of the rectangle is ',width_height)
        if(float(width_height)>threshold_score):
               print("stop_sign_detected")
        else:
               print("stop sign is still far")
        subprocess.call(['./remove_file.sh'])#print('file removed')
        flag=1       
        end_time = time.time()
        net_time = end_time - start_time
        print(net_time)
    except:
        
        if flag !=0 :
            print("stop_sign processing is being done..please wait")
            flag=0    

