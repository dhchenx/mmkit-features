from mmkfeatures.graph.gan_node import GanNode
'''
    Create a GAN-based node that self-generates an instance of image from the model
'''
if __name__ == "__main__":
    root_path="models/gan_calendula"
    data_path="../../datasets/Flower_MNIST_JPG/calendula/0003.jpg"
    gnode=GanNode(root_path=root_path,  dim=64)
    # gnode.create_samples_from(src_img_path=data_path)
    # gnode.build_model(iterations=1000)
    gnode.export_performance()
    gnode.generate(n_samples=5,show=True)


