from mmkfeatures.graph.gan_node import GanNode
from mmkfeatures.graph.cnn_node import CnnNode
from PIL import Image
import numpy as np


'''
    link the model between GAN node and CNN nodes
'''
if __name__ == "__main__":
    root_path="models/gan_calendula"
    data_path="../../datasets/Flower_MNIST_JPG/calendula/0003.jpg"
    gnode=GanNode(root_path=root_path,  dim=64)
    # gnode.create_samples_from(src_img_path=data_path)
    # gnode.build_model(iterations=1000)
    # gnode.export_performance()
    X=gnode.generate(n_samples=10)
    img0=X[0]

    cnn_node = CnnNode(root_path="models", name="flower_cnn")
    # cnn_node.build(n_epochs=10, train_dir="data/Flower_MNIST", use_file_names=True)
    # clear
    # X = Image.open("data/Flower_MNIST/rose/0006.jpg")
    # X = np.array(X)
    img = np.resize(img0, (64, 64, 3))
    img = np.reshape(img, (img.shape[0], img.shape[1], 3))
    print(X.shape)
    cnn_node.predict(img, channels=3,show_fig=True)
