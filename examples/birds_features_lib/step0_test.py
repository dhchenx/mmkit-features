
path=r"datasets\\CUB_200_2011\\images\\001.Black_footed_Albatross\\Black_Footed_Albatross_0046_18.jpg"
# path="datasets/Black_Footed_Albatross_0046_18.jpg"
# path=path.replace("/","\\")
print(path)
import cv2

import os
if os.path.exists(path):
    print("file exists!")

image=cv2.imread(path)
# print(image)
# cv2.imshow("test",image)
# cv2.waitKey(0)

bbox=(60,27,325,304)

# Window name in which image is displayed
window_name = 'Image'

# Start coordinate, here (5, 5)
# represents the top left corner of rectangle
start_point = (bbox[0], bbox[1])

# Ending coordinate, here (220, 220)
# represents the bottom right corner of rectangle
end_point = (bbox[0]+bbox[2], bbox[1]+bbox[3])

# Blue color in BGR
color = (255, 0, 0)

# Line thickness of 2 px
thickness = 2

# Using cv2.rectangle() method
# Draw a rectangle with blue line borders of thickness of 2 px
image = cv2.rectangle(image, start_point, end_point, color, thickness)

raw_image=image

point=(186,45)
width=20
part_rect_start=(point[0]-width,point[1]-width)
part_rect_end=(point[0]+width,point[1]+width)
# image=cv2.circle(image,point,radius=10,color=color,thickness=thickness)
part_color=(0,255,0)
image = cv2.rectangle(image, part_rect_start, part_rect_end, part_color, thickness-1)

# Displaying the image
cv2.imshow(window_name, image)

cv2.waitKey(0)

image=cv2.imread(path)

box_image=image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]

cv2.imshow(window_name,box_image)

cv2.waitKey(0)

image=cv2.imread(path)

part_image=image[point[1]-width:point[1]+width,point[0]-width:point[0]+width]

cv2.imshow(window_name,part_image)

cv2.waitKey(0)

from mmkfeatures.image.color_descriptor import ColorDescriptor
cd = ColorDescriptor((8, 12, 3))
print(cd.describe(part_image))


