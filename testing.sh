#!/bin/bash
while True
do
scp pi@raspberrypi.local:/home/pi/PiCar-2019/webcam9.png ~/Stop-Sign-Detection/
source /Applications/anaconda3/bin/activate
conda activate base
cd /Users/kopadava/Stop-Sign-Detection/
python stop_sign.py -p stopPrototype.png -i webcam9.png
echo "calling done"
scp /Users/kopadava/Stop-Sign-Detection/width_height.txt pi@raspberrypi.local:/home/pi/PiCar-2019/
rm -f "webcam9.png"
rm -f "width_height.txt"
echo "image and file are removed"
done
