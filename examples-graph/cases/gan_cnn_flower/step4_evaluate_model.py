from mmkfeatures.graph.gan_node import GanNode
from mmkfeatures.graph.cnn_node import CnnNode
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

'''
    evaluate the model based on bridging the input and output of models
'''
if __name__ == "__main__":
    label="calendula"
    root_path="models/gan_calendula"
    data_path=f"../../datasets/Flower_MNIST_JPG/{label}/0003.jpg"
    gnode=GanNode(root_path=root_path,  dim=64)
    # gnode.create_samples_from(src_img_path=data_path)
    # gnode.build_model(iterations=1000)
    # gnode.export_performance()
    X=gnode.generate(n_samples=10)

    cnn_node = CnnNode(root_path="models", name="flower_cnn")
    # cnn_node.build(n_epochs=10, train_dir="data/Flower_MNIST", use_file_names=True)
    list_result=[]
    print("len(X) = ",len(X))
    num_correct=0
    num_total=len(X)
    for img in X:
        # X = Image.open("data/Flower_MNIST/rose/0006.jpg")
        # X = np.array(X)
        # img = np.resize(img, (64, 64, 3))
        # img = np.reshape(img, (img.shape[0], img.shape[1], 3))
        predicted=cnn_node.predict(img, channels=3,show_fig=False)
        if label==predicted:
            num_correct+=1

        list_result.append(predicted)

    # results
    print("the results: ")
    for x in list_result:
        print(x)
    print("correct rate: ",round(num_correct*1.0/num_total,4))

