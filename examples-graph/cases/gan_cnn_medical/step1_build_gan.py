from mmkfeatures.graph.gan_node import GanNode
from mmkfeatures.graph.cgan_node import CGanNode
from mmkfeatures.graph.cnn_node import CnnNode
from PIL import Image
import numpy as np
import cv2
from keras.preprocessing import image
import os

if __name__=="__main__":
    label="HeadCT"
    root_path = f"models/{label}"
    gnode = GanNode(root_path=root_path, dim=64)
    # gnode.create_samples_from(src_img_path="../../datasets/Medical_MNIST/HeadCT/000000.jpeg",use_gray=True)
    # gnode.build_model(iterations=1000,use_gray=True)
    # gnode.export_performance()

    X=gnode.generate(n_samples=10) # generated image
    img0=X[0]
    print(X[9])

    # img0=np.array(X[0])
    img0=np.resize(img0,(64,64,1))
    # img0 = np.reshape(img0, (img0.shape[0], img0.shape[1], 1))
    img = image.array_to_img(img0 * 255., scale=False)
    img.save('savedimage.png')

    cnn_node = CnnNode(root_path="models", name="medical_cnn")

    # cnn_node.build(n_epochs=10,train_dir="../../datasets/Medical_MNIST")
    # clear
    img0 = Image.open("savedimage.png")
    img0 = np.array(img0)
    img0 = np.reshape(img0, (img0.shape[0], img0.shape[1], 1))

    predicted=cnn_node.predict(img0,show_fig=False)

