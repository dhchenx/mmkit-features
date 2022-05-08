import cv2

image_path=r"datasets/all_data/train/radiology/images/PMC4083729_AMHSR-4-14-g002.jpg"

image=cv2.imread(image_path)
print(image.shape)
cv2.imshow("title",image)
cv2.waitKey(0)