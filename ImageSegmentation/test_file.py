import cv2
import numpy as np

#ann_img = np.zeros((30,30,3)).astype('uint8')
#ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

#cv2.imwrite( "1.jpg" ,ann_img )
cv2.namedWindow('1', cv2.WINDOW_NORMAL)
file = open('IITB.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(image, file)
file.close()