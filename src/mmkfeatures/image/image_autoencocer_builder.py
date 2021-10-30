import random
import pickle
import matplotlib
matplotlib.use("Agg")
from mmkfeatures.image.conv_autoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.models import load_model
from imutils import build_montages

class ImageAutoEncoderBuilder():

    def __init__(self,epochs=20,init_lr=1e-3,bs=32,img_size=(64,64),channel=3,split=0.8):
        self.EPOCHS = epochs
        self.INIT_LR = init_lr
        self.BS = bs
        self.img_size = img_size
        self.channel = channel
        self.split = split
        self.H=None
        self.autoencoder=None

    def create_datasets(self, data, is_shuffle=False):
        list_cids = list(data.keys())
        if is_shuffle:
            random.shuffle(list_cids)
        Xs=[]
        Ids=[]
        for cid in tqdm(list_cids):
            item = data[cid]
            imgs = item["objects"]
            for img_id in imgs:
                # print(image)
                img = imgs[img_id][()]
                resized_img = cv2.resize(img, dsize=self.img_size)
                Xs.append(resized_img)
                Ids.append(cid)
        Xs = np.array(Xs)
        Xs = Xs.astype("float32") / 255.0
        return Ids,Xs

    def create_train_test_ids(self,data,total_sample_size=-1, is_shuffle=False):
        print("[INFO] loading dataset...")
        trainX = []
        testX = []

        list_cids = list(data.keys())

        if is_shuffle:
            random.shuffle(list_cids)

        if total_sample_size != -1:
            list_cids = list_cids[:total_sample_size]

        if is_shuffle:
            random.shuffle(list_cids)

        total_num = len(list_cids)
        train_num = int(total_num * self.split)

        test_num = total_num - train_num

        list_cids_train = list_cids[:train_num]
        list_cids_test = list_cids[train_num:]

        for cid in tqdm(list_cids_train):
            item = data[cid]
            imgs = item["objects"]
            for img_id in imgs:
                # print(image)
                img = imgs[img_id][()]
                resized_img = cv2.resize(img, dsize=self.img_size)
                trainX.append(resized_img)

        for cid in tqdm(list_cids_test):
            item = data[cid]
            imgs = item["objects"]
            for img_id in imgs:
                # print(image)
                img = imgs[img_id][()]
                resized_img = cv2.resize(img, dsize=self.img_size)
                testX.append(resized_img)

        trainX = np.array(trainX)
        testX = np.array(testX)
        trainX = trainX.astype("float32") / 255.0
        testX = testX.astype("float32") / 255.0
        return trainX,testX,list_cids_train,list_cids_test

    def train(self,trainX,testX, output_model_path=""):

        # construct our convolutional autoencoder
        print("[INFO] building autoencoder...")
        self.autoencoder = ConvAutoencoder.build(self.img_size[0], self.img_size[1], self.channel)
        opt = Adam(lr=self.INIT_LR, decay=self.INIT_LR / self.EPOCHS)
        self.autoencoder.compile(loss="mse", optimizer=opt)
        # train the convolutional autoencoder
        self.H = self.autoencoder.fit(
            trainX, trainX,
            validation_data=(testX, testX),
            epochs=self.EPOCHS,
            batch_size=self.BS)

        print("[INFO] saving autoencoder...")
        if output_model_path!="":
            self.autoencoder.save(output_model_path, save_format="h5")
        return self.H

    def visualize_predictions(self,decoded, gt, samples=10):
        # initialize our list of output images
        outputs = None
        # loop over our number of output samples
        for i in range(0, samples):
            # grab the original image and reconstructed image
            original = (gt[i] * 255).astype("uint8")
            recon = (decoded[i] * 255).astype("uint8")
            # stack the original and reconstructed image side-by-side
            output = np.hstack([original, recon])
            # if the outputs array is empty, initialize it as the current
            # side-by-side image display
            if outputs is None:
                outputs = output
            # otherwise, vertically stack the outputs
            else:
                outputs = np.vstack([outputs, output])
        # return the output images
        return outputs

    def predict(self,testX,_vis="", _plot=""):
        print("[INFO] making predictions...")
        decoded = self.autoencoder.predict(testX)
        if _vis != "":
            vis = self.visualize_predictions(decoded, testX)
            cv2.imwrite(_vis, vis)
        # construct a plot that plots and saves the training history
        if _plot != "":
            N = np.arange(0, self.EPOCHS)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H.history["loss"], label="train_loss")
            plt.plot(N, self.H.history["val_loss"], label="val_loss")
            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig(_plot)
        return decoded

    def predict_image(self,testX,samples=10,_vis="",_plot=""):
        decoded=self.predict(testX,_vis=_vis,_plot=_plot)
        # initialize our list of output images
        outputs = []
        # loop over our number of output samples
        for i in range(0, samples):
            # grab the original image and reconstructed image
            original = (testX[i] * 255).astype("uint8")
            recon = (decoded[i] * 255).astype("uint8")
            # stack the original and reconstructed image side-by-side
            output = [original,recon]
            # if the outputs array is empty, initialize it as the current
            # side-by-side image display
            outputs.append(output)

        # return the output images
        return outputs

    def load_model(self,model_file):
        print("[INFO] loading autoencoder model...")
        self.autoencoder = load_model(model_file)
        return self.autoencoder

    def create_index(self,Ids,Xs,index_file):
        # _model = "output/autoencoder.h5"
        # _index = "output/index.pickle"

        # load the MNIST dataset
        print("[INFO] loading data...")

        # create the encoder model which consists of *just* the encoder
        # portion of the autoencoder
        self.encoder = Model(inputs=self.autoencoder.input,
                        outputs=self.autoencoder.get_layer("encoded").output)
        # quantify the contents of our input images using the encoder
        print("[INFO] encoding images...")
        features = self.encoder.predict(Xs)

        # construct a dictionary that maps the index of the MNIST training
        # image to its corresponding latent-space representation

        data = {"indexes": Ids, "features": features}
        # write the data dictionary to disk
        print("[INFO] saving index...")
        f = open(index_file, "wb")
        f.write(pickle.dumps(data))
        f.close()
        return self.encoder

    def euclidean(self,a, b):
        # compute and return the euclidean distance between two vectors
        return np.linalg.norm(a - b)

    def perform_search(self,queryFeatures, index, maxResults=64):
        # initialize our list of results
        results = []
        # loop over our index
        for i in range(0, len(index["features"])):
            # compute the euclidean distance between our query features
            # and the features for the current image in our index, then
            # update our results list with a 2-tuple consisting of the
            # computed distance and the index of the image
            d = self.euclidean(queryFeatures, index["features"][i])
            results.append((d, i))
        # sort the results and grab the top ones
        results = sorted(results)[:maxResults]
        # return the list of results
        return results

    def search(self,index_file, queryXs, display=True,  Xs=None,display_num=5, max_result=10):
        index = pickle.loads(open(index_file, "rb").read())
        Ids = index["indexes"]

        encoder = Model(inputs=self.autoencoder.input,
                        outputs=self.autoencoder.get_layer("encoded").output)

        features = encoder.predict(queryXs)
        list_results=[]
        count=0
        for idx,f in enumerate(features):
            count+=1
            results = self.perform_search(f, index, maxResults=max_result)
            images = []
            ids=[]
            # loop over the results
            for (d, j) in results:
                # grab the result image, convert it back to the range
                # [0, 255], and then update the images list
                # image = np.dstack([image] * 3)
                if display and count<=display_num:
                    image = (Xs[j] * 255).astype("uint8")
                    images.append(image)
                ids.append(Ids[j])
            if display and count<=display_num:
                # display the query image
                query = (queryXs[idx] * 255).astype("uint8")
                cv2.imshow("Query", query)
                # build a montage from the results and display it
                montage = build_montages(images, (self.img_size[0], self.img_size[1]), (15, 15))[0]
                cv2.imshow("Results", montage)
                cv2.waitKey(0)
            list_results.append([ids,images])
        return list_results


