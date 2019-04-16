"""
Take one picture and save in Pi home directory
"""
import picamera

camera = picamera.PiCamera()
camera.resolution = (420, 368)
camera.brightness = 60
camera.start_preview()

img_sav_name = '~/pi/' + 'image.jpeg' 
img = camera.capture(img_sav_name)
camera.stop_preview()
img_matrix = cv2.imread(img_sav_name,0)