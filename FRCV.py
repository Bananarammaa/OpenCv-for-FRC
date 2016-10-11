import cv
import cv2
import numpy as np
from networktables import NetworkTable
import logging
logging.basicConfig(level=logging.DEBUG)

from datetime import datetime
import os
import sys
import time
# network table setup

#commented out for testing
NetworkTable.setIPAddress("localhost")#127.0.0.1 with tester program
NetworkTable.setClientMode()
NetworkTable.initialize()
sd = NetworkTable.getTable("SmartDashboard")

# VideoCapture webcam id=0,1,2,3...
vc = cv2.VideoCapture(0)

# try to get the first frame
if vc.isOpened():
    rval, src = vc.read()
else:
    rval = False

'''
vc.set(cv2.CAP_PROP_FRAME_WIDTH,320)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
#vc.set(cv2.CAP_PROP_FPS,30)
vc.set(cv2.CAP_PROP_EXPOSURE, -8.0)
'''
# Set up FPS list and iterator
times = [0] * 25
time_idx = 0
time_start = time.time()
camfps=0
i = 0
j = 0
found = False
#Loop to process video
while rval:
    # Compute FPS information
    time_end = time.time()
    times[time_idx] = time_end - time_start
    time_idx += 1
    if time_idx >= len(times):
    	camfps = 1/(sum(times)/len(times))
    	time_idx = 0
    if time_idx > 0 and time_idx % 5 == 0:
    	camfps = 1/(sum(times)/len(times))
    time_start = time_end
    #value to alter
    thrval = 120

    #rval, src = vc.read()#camera feed
    src = cv2.imread("/Users/Joseph/Desktop/tower.png")#image file

    # convert the image to grayscale and threshold it
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(src, cv2.COLOR_BGR2HLS)
    #cv2.imshow("gray",gray)

    thr=gray
    thr = cv2.inRange(gray,120,150,thr)
    #cv2.inRange(thr,(63,55,168), (96,161,255),thr)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    #thr = cv2.cvtColor(thr, cv2.COLOR_BGR2GRAY)
    #cv2.inRange(thr,120,150,thr)
    #cv2.imshow("thr",thr)

    lower_teal = np.array([0,158,87])
    upper_teal = np.array([96,255,175])

    lower_red = np.array([89,128,105])
    upper_red = np.array([149,255,223])

    mask = cv2.inRange(hsv, lower_teal, upper_teal)# for tower'''
    '''mask = cv2.inRange(hsv, lower_red, upper_red)#for testing'''
    #cv2.imshow("hsv", mask)
    #blur = cv2.blur(mask,(5,5))
    edges = cv2.Canny(mask, 100,200)
    #cv2.imshow("edges",edges)
    try:
        # find the contours and keep the largest one
        #(_,cnts, _) = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #used to be thr
        cnts,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(src,cnts,-1,(255,105,180),3)
        cnt = cnts[i]#11
        area = cv2.contourArea(cnt)
        while area < 100:
            print 'contour', i, 'is ', area, 'big'
            i = i+1
            cnt = cnts[i]
            area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(src, 'cx: '+str(cx),(20,100), font ,1, (255, 255, 0), 2)
        cv2.putText(src, 'cy: '+str(cy),(20,70), font ,1, (255, 255, 0), 2)
        cv2.circle(src, (cx,cy), 10, (255,255,255), -1)

        print ('cx is...', cx)
        print ('cy is...', cy)




        #DOESNT WORK
        #c = max(cnts,key = cv2.contourArea, default=1)   # draw bounding recangle

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(src,(x,y),(x+w,y+h),(128,255,0),2)

        '''
        rect = cv2.minAreaRect(cnt)
        print rect
        box = cv2.boxPoints(rect)
        print('boop')
        box = np.int0(box)
        cv2.drawContours(src,[box],0,(0,255,255),2)
        '''


        # send to network tables
        print('SendingX... ', (x+w/2))
        sd.putNumber('COG_X', (x+w/2))
        print('SendingY... ', (y+h/2))
        sd.putNumber('COG_Y', (y+h/2))

        '''
        moments = cv.Moments(cv.fromarray(edges),i)
        area = cv.GetCentralMoment(moments, 0, 0)

        x = cv.GetSpatialMoment(moments, 1, 0)/area
        y = cv.GetSpatialMoment(moments, 0, 1)/area

        #create an overlay to mark the center of the tracked object
        overlay = cv.CreateImage(cv.GetSize(cv.fromarray(src)), 8, 3)

        cv.Circle(cv.fromarray(src), (int(x), int(y)), 2, (255, 255, 255), 20)
        print ('area is...', int(area))
        print ('x is...', int(x))
        print ('y is...', int(y))
        print ((x/320)*2)-1
        cv2.imshow("src", src)
        '''
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)

        for j in range(defects.shape[0]):
            s,e,f,d = defects[j,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(src,start,end,[0,255,0],2)
        key = cv2.waitKey(20)
        if key == 27:
            break
    except:
        print('an error has occured')

    # display image
    cv2.imshow("src", src)

    # escape key: ESC
    key = cv2.waitKey(20)
    if key == 27:
        break

# close the window and releases camera
cv2.destroyAllWindows()
vc.release()
