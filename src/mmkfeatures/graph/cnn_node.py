import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.losses import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CnnNode:

    def __init__(self,root_path,name):
        # load datasets
        self.name = name
        self.root_path=root_path
        self.model_root_path=self.root_path+"/"+name
        self.data_path=self.model_root_path+"/data"
        self.model_path=self.model_root_path+f"/{name}.h5"
        if not os.path.exists(self.model_root_path):
            os.mkdir(self.model_root_path)


    # Converts numbers to 6 figure strings
    # e.g. '24' to '000024', '357' to '000357'
    def to_6sf(self,n):
        new = str(n)
        new = '0' * (6 - len(new)) + new
        return new

    def build(self,n_epochs=10,train_dir="",ext=".jpeg",use_file_names=False):
        if train_dir=="":
            train_dir=self.data_path
        _, labels, _ = next(os.walk(train_dir))
        pickle.dump(labels,open(self.model_root_path+"/labels.pickle","wb"))
        print('Labels :', labels)
        x_all = []
        y_all = []
        for x in labels:

            if use_file_names:
                for file in os.listdir(train_dir + "/" + x ):
                    X = Image.open(train_dir + "/" + x + '/' + file)
                    X = np.array(X)
                    X=np.resize(X,(64,64,3))
                    X = np.reshape(X, (X.shape[0], X.shape[1], 3))
                    x_all.append(X)
                    y_all.append(x)
            else:
                num_images = len(os.listdir(train_dir + "/" + x + '/'))
                for i in range(num_images):
                    X = Image.open(train_dir + "/" + x + '/' + self.to_6sf(i) + ext)
                    X = np.array(X)
                    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                    x_all.append(X)
                    y_all.append(x)
        x_all = np.array(x_all)
        y_all = np.array(y_all)

        print('Total Samples :', len(x_all))

        ## preprocess datasets

        # Prepare Encoder-Decoder Dict
        encode_y = dict()
        decode_y = dict()
        for x in enumerate(labels):
            encode_y[x[1]] = x[0]
            decode_y[x[0]] = x[1]
        print('Encoder :', encode_y)
        print('Decoder :', decode_y)

        # Apply Encoder dict
        for x in range(len(x_all)):
            y_all[x] = encode_y[y_all[x]]
        y_all = np.array(y_all, dtype=np.int16)

        ## Split to Train & Test
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, shuffle=True)
        print('Total Train Samples :', len(x_train))
        print('Total Test Samples :', len(x_test))

        # Model
        input_shape = x_all[0].shape
        print('Image Shape :', input_shape)

        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(40, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(164, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(42, activation='relu'))
        model.add(Dense(len(labels), activation='softmax'))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        ## train model
        checkpoint = ModelCheckpoint(self.model_path)

        history = model.fit(x_all, y_all, epochs=n_epochs,
                  validation_data=(x_test, y_test), callbacks=[checkpoint])
        print(history.history)
        pickle.dump(history.history,open(self.model_root_path+"/history.pickle","wb"))

        # labels = pickle.load(open(self.model_root_path + "/labels.pickle", "rb"))

        # Load pretrained model
        model = load_model(self.model_path)
        loss, acc = model.evaluate(x_test, y_test)
        print('Test Loss :', loss)
        print('Test Accuracy :', acc)

    def show_history(self):
        if os.path.exists(self.model_root_path+"/history.pickle"):
            history=pickle.load(open(self.model_root_path+"/history.pickle","rb"))

            print(history)
            metrics=list(history.keys())
            c=len(history["accuracy"])
            f_out=open(self.model_root_path+"/history.csv","w",encoding='utf-8')
            print("\t".join(metrics))
            f_out.write("\t".join(metrics)+"\n")
            for i in range(c):
                line=""
                for l in metrics:
                    line+=str(history[l][i])+"\t"
                print(line)
                f_out.write(line+"\n")
            f_out.close()


    def predict(self,img,show_fig=False,channels=1):
        labels = pickle.load(open(self.model_root_path + "/labels.pickle", "rb"))
        # Prepare Encoder-Decoder Dict
        encode_y = dict()
        decode_y = dict()
        for x in enumerate(labels):
            encode_y[x[1]] = x[0]
            decode_y[x[0]] = x[1]
        print('Encoder :', encode_y)
        print('Decoder :', decode_y)

        # Load pretrained model
        model = load_model(self.model_path)

        # Predict
        # IMAGE_INDEX = 100
        # print("x_test.shape", x_test.shape)
        # img = x_test[IMAGE_INDEX]
        if show_fig:
            plt.title("origin")
            plt.imshow(img.squeeze())
            plt.show()

        # print('Actual :', decode_y[y_test[IMAGE_INDEX]])

        img = np.reshape(img, (1, 64, 64, channels))
        predicted_obj = model.predict(img)

        predicted_obj1 = np.argmax(predicted_obj)
        predicted_label = decode_y[predicted_obj1]
        print('Predicted :', predicted_label)
        return predicted_label

