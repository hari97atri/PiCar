"""
Program to test motor movement
"""
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(17,GPIO.OUT)
GPIO.setup(18,GPIO.OUT)
GPIO.setup(22,GPIO.OUT)
GPIO.setup(23,GPIO.OUT)

for i in range(10):
    
    GPIO.output(17,GPIO.HIGH)
    GPIO.output(18,GPIO.LOW)
    GPIO.output(22,GPIO.HIGH)
    GPIO.output(23,GPIO.LOW)
