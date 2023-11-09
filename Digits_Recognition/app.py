#                        '''HANDWRITTEN DIGITS RECOGNITION'''

import pygame, sys
from pygame.locals import *
from pygame import image
import numpy as np
from keras.models import load_model
import cv2

'''WINDOW SIZE'''
x=640
y=480

i_count=0
IMAGESAVE=True
BOUNDARYINC=5
iswriting=False

num_xcord=[]
num_ycord=[]

W=(255,255,255)
B=(0,0,0)
R=(255,0,0)


pygame.init()
'''MODEL'''
model=load_model('bestmodel.h5')
PREDICT=True

FONT=pygame.font.Font("freesansbold.ttf",30)
labels=[0,1,2,3,4,5,6,7,8,9]

DISPLAYSURF=pygame.display.set_mode((x,y))
pygame.display.set_caption('Digit Recognizer')

while True:

    for event in pygame.event.get():
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type==MOUSEMOTION and iswriting:
            xcord,ycord=event.pos
            pygame.draw.circle(DISPLAYSURF,W,(xcord,ycord),4,0)

            num_xcord.append(xcord)
            num_ycord.append(ycord)
        '''WRITTING'''
        if event.type==MOUSEBUTTONDOWN:
            iswriting=True

        if event.type==MOUSEBUTTONUP:
            iswriting=False
            num_xcord=sorted(num_xcord)
            num_ycord=sorted(num_ycord)

            rect_min_x,rect_max_x=max(num_xcord[0]-BOUNDARYINC,0),min(x,num_xcord[-1]+BOUNDARYINC)
            rect_min_y,rect_max_y=max(num_ycord[0]-BOUNDARYINC,0),min(y,num_ycord[-1]+BOUNDARYINC)

            num_xcord=[]
            num_ycord=[]
            '''READING THE DISPLAY'''
            img_arr=np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if 0:
                cv2.imwrite("help.png")
                i_count+=1
            if PREDICT:
                image=cv2.resize(img_arr,(28,28))
                image=np.pad(image,(10,10),"constant",constant_values=0)
                image=cv2.resize(image,(28,28))/255
                print(model.predict(image.reshape(1,28,28,1)))
                '''PREDICTING'''
                label=str(labels[np.argmax(model.predict(image.reshape(1,28,28,1)))])

                textSurface=FONT.render(label,True,R,W)

                DISPLAYSURF.blit(textSurface,(rect_min_x,rect_max_y))
            if event.type == pygame.KEYDOWN:
                if event.unicode == 'n':
                    DISPLAYSURF.fill(B)
        pygame.display.update()
