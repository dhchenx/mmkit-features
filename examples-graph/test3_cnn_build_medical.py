from mmkfeatures.graph.cnn_node import CnnNode
from PIL import Image
import numpy as np

if __name__=="__main__":
    cnn_node=CnnNode(root_path="models",name="medical_cnn")
    cnn_node.build(n_epochs=10,train_dir="datasets/Medical_MNIST")
    # clear
    X = Image.open("datasets/Medical_MNIST/ChestCT/000000.jpeg")
    X = np.array(X)
    img = np.reshape(X, (X.shape[0], X.shape[1], 1))
    cnn_node.predict(img)

