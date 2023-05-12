from mmkfeatures.graph.cnn_node import CnnNode
from PIL import Image
import numpy as np

if __name__=="__main__":
    cnn_node=CnnNode(root_path="models",name="flower_cnn")
    cnn_node.build(n_epochs=1000,train_dir="datasets/Flower_MNIST",use_file_names=True)
    cnn_node.show_history()
    # clear
    X = Image.open("datasets/Flower_MNIST/rose/0006.jpg")
    X = np.array(X)
    img = np.resize(X,(64,64,3))
    img = np.reshape(img, (img.shape[0], img.shape[1], 3))
    print(X.shape)
    cnn_node.predict(img,channels=3)
