import matplotlib.pyplot as plt
import h5py
import cv2
import tensorflow as tf
from keras.layers import Conv2D, Dense, LeakyReLU, Dropout, Input
from keras.layers import Reshape, Conv2DTranspose, Flatten
from keras.models import Model
from keras import optimizers
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
import numpy as np
import os
from tqdm import tqdm
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GanNode:

    def __init__(self,root_path,dim=32):
        self.d_loss=[]
        self.a_loss=[]
        self.root_path=root_path
        self.dim=dim
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)

    def create_samples_from(self,src_img_path,sample_name="sample",num_samples=12000,use_gray=False):
        print("Creating samples from a picture...")
        datagen = ImageDataGenerator(rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')
        if use_gray:
            img = load_img(src_img_path,grayscale=True)
        else:
            img = load_img(src_img_path)
        # img = img.convert('L')
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)



        save_to_dir=self.root_path+"/data"

        if not os.path.exists(save_to_dir):
            os.mkdir(save_to_dir)

        i = 0
        for batch in datagen.flow(x, batch_size=200,
                                  save_to_dir=save_to_dir, save_prefix=sample_name, save_format='jpg'):
            i += 1
            if i > num_samples:
                break

    def get_files(self,file_dir):
        cats = []

        label_cats = []
        # dogs = []
        # label_dogs = []

        for file in os.listdir(file_dir ):
            cats.append(file_dir + '/' + file)
            label_cats.append(0)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
        # for file in os.listdir(file_dir + '/1'):
        #    dogs.append(file_dir + '/1' + '/' + file)
        #    label_dogs.append(1)

        # 把cat和dog合起来组成一个list（img和lab）
        # image_list = np.hstack((cats, dogs))
        # label_list = np.hstack((label_cats, label_dogs))
        image_list = cats
        label_list = label_cats

        # 利用shuffle打乱顺序
        temp = np.array([image_list, label_list])
        temp = temp.transpose()
        np.random.shuffle(temp)

        # 从打乱的temp中再取出list（img和lab）
        image_list = list(temp[:, 0])
        label_list = list(temp[:, 1])
        label_list = [int(i) for i in label_list]

        return image_list, label_list
        # 返回两个list 分别为图片文件名及其标签 顺序已被打乱

    def create_dataset_gray(self,train_dir,out_path):

        # train_dir = '../data_preprocess/train'
        image_list, label_list = self.get_files(train_dir)

        # print(len(image_list))
        # print(len(label_list))

        # 450为数据长度的20%
        percent20 = int(len(image_list) * 0.2 )
        dim = self.dim
        Train_image = np.random.rand(len(image_list) - percent20, dim, dim).astype('float32')
        Train_label = np.random.rand(len(image_list) - percent20, 1).astype('float32')

        Test_image = np.random.rand(percent20, dim, dim).astype('float32')
        Test_label = np.random.rand(percent20, 1).astype('float32')

        print("train len: ",len(Train_image))
        print("test len: ", len(Test_image))
        for i in range(len(image_list) - percent20):
            img = plt.imread(image_list[i])
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)

            Train_image[i] = np.array(shrink_img)
            Train_label[i] = np.array(label_list[i])

        for i in range(len(image_list) - percent20, len(image_list)):
            img = plt.imread(image_list[i])

            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)

            Test_image[i + percent20 - len(image_list)] = np.array(shrink_img)
            Test_label[i + percent20 - len(image_list)] = np.array(label_list[i])

        # Create a new file
        f = h5py.File(out_path, 'w')
        f.create_dataset('X_train', data=Train_image)
        f.create_dataset('y_train', data=Train_label)
        f.create_dataset('X_test', data=Test_image)
        f.create_dataset('y_test', data=Test_label)
        f.close()

    def create_dataset(self,train_dir,out_path,use_gray=False):

        # train_dir = '../data_preprocess/train'
        image_list, label_list = self.get_files(train_dir)

        # print(len(image_list))
        # print(len(label_list))

        # 450为数据长度的20%
        percent20 = int(len(image_list) * 0.2 )
        dim = self.dim
        Train_image = np.random.rand(len(image_list) - percent20, dim, dim, 3).astype('float32')
        Train_label = np.random.rand(len(image_list) - percent20, 1).astype('float32')

        Test_image = np.random.rand(percent20, dim, dim, 3).astype('float32')
        Test_label = np.random.rand(percent20, 1).astype('float32')

        print("train len: ",len(Train_image))
        print("test len: ", len(Test_image))
        for i in range(len(image_list) - percent20):
            img = plt.imread(image_list[i])
            if use_gray:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
            else:
                shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
            Train_image[i] = np.array(shrink_img)
            Train_label[i] = np.array(label_list[i])

        for i in range(len(image_list) - percent20, len(image_list)):
            img = plt.imread(image_list[i])
            if use_gray:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
            else:
                shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
            Test_image[i + percent20 - len(image_list)] = np.array(shrink_img)
            Test_label[i + percent20 - len(image_list)] = np.array(label_list[i])

        # Create a new file
        f = h5py.File(out_path, 'w')
        f.create_dataset('X_train', data=Train_image)
        f.create_dataset('y_train', data=Train_label)
        f.create_dataset('X_test', data=Test_image)
        f.create_dataset('y_test', data=Test_label)
        f.close()

    def build64(self,data_path,model_path,save_dir,iterations=1000,batch_size=64,height=32,width=32,latent_dim=100,channels=3):

        """设置相关参数"""
        # 潜变量维度
        # latent_dim = 100
        # 输入像素维度
        # height = 64
        #width = 64
        # channels = 3

        # 搭建生成器网络（Model方式）
        generator_input = Input(shape=(latent_dim,))
        print("shape1", generator_input.shape)
        x = Dense(128 * 32 * 32)(generator_input)
        print("shape2", x.shape)
        x = LeakyReLU()(x)
        print("shape3", x.shape)
        x = Reshape((32, 32, 128))(x)
        print("shape4", x.shape)

        # IN: 16*16*128  OUT: 16*16*256
        x = Conv2D(256, 5, padding='same')(x)
        print("shape5", x.shape)
        x = LeakyReLU()(x)
        print("shape6", x.shape)

        # IN: 16*16*256  OUT: 32*32*256
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)
        print("shape7", x.shape)

        # 通过反卷积操作，把维度变得和图片一样大了，就别用反卷积了
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        print("shape8", x.shape)

        # 把通道变回来
        x = Conv2D(channels, 7, activation='tanh', padding='same')(x)
        print("shape9", x.shape)
        generator = Model(generator_input, x)
        # generator.summary()

        # 搭建判别器，判别器就真的是一个卷积神经网络了
        discriminator_input = Input(shape=(height, width, channels))
        # IN:32*32*3   OUT: 30*30*128
        x = Conv2D(128, 3)(discriminator_input)
        x = LeakyReLU()(x)

        # IN: 30*30*128  OUT:14*14*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:14*14*128   OUT:6*6*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:6*6*128  OUT:2*2*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # 展开成512个神经元
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(1, activation='sigmoid')(x)

        discriminator = Model(discriminator_input, x)
        # discriminator.summary()

        discriminator_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0008,
                                                     clipvalue=1.0,
                                                     decay=1e-8)

        discriminator.compile(optimizer=discriminator_optimizer,
                              loss='binary_crossentropy')

        # 将上面两个连起来组成生成对抗网络
        # 将判别器参数设置为不可训练
        discriminator.trainable = False

        gan_input = Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        # 搭建对抗网络
        gan = Model(gan_input, gan_output)
        gan_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0004,
                                           clipvalue=1.0,
                                           decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

        ## 下面导入数据集进行训练，数据集选用的是Keras自带的CIFAR-10数据集
        import os
        from keras.preprocessing import image

        # 加载数据集（手写数字的数据集也可以这样加载）
        # (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
        # 指定青蛙图像（编号为6）
        # x_train = x_train[y_train.flatten() == 6]
        # Load hdf5 dataset

        import h5py
        train_dataset = h5py.File(data_path, 'r')
        train_set_x_orig = np.array(train_dataset['X_train'][:])  # your train set features
        train_set_y_orig = np.array(train_dataset['y_train'][:])  # your train set labels
        test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
        test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
        train_dataset.close()

        print(train_set_x_orig.shape)
        print(train_set_y_orig.shape)

        print(train_set_x_orig.max())
        print(train_set_x_orig.min())

        print(test_set_x_orig.shape)
        print(test_set_y_orig.shape)
        x_train = train_set_x_orig

        print("data size:", x_train.shape)
        x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

        print("data size(adjusted):", x_train.shape)

        # 下面开始训练
        # iterations = 10000
        # batch_size = 32
        #save_dir = 'image64'

        start = 0
        for step in range(iterations):

            # 先通过噪声生成64张伪造图片
            noise = np.random.normal(size=(batch_size, latent_dim))
            images_fake = generator.predict(noise)

            # 从真实图片中抽64张
            end = start + batch_size
            images_train = x_train[start:end]

            # 两者混合作为训练集，并打上标签
            x = np.concatenate([images_fake, images_train])
            y = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])

            # 向标签中添加噪声
            y += 0.05 * np.random.random(y.shape)

            # 训练判别器
            d_loss = discriminator.train_on_batch(x, y)

            # 训练生成对抗网络
            noise = np.random.normal(size=(batch_size, latent_dim))
            labels = np.ones((batch_size, 1))

            # 通过gan模型来训练生成器模型，冻结判别器模型权重
            a_loss = gan.train_on_batch(noise, labels)

            start += batch_size
            if start > len(x_train) - batch_size:
                start = 0

            # 每500步绘图并保存
            if step % 100 == 0:
                generator.save(model_path)
            # if step % 10 == 0:
            print(f"Iteration={step + 1}\tDLoss={d_loss}\tALoss={a_loss}")
            # print('discriminator loss:', d_loss)
            # print('adversarial loss:', a_loss)
            self.d_loss.append(d_loss)
            self.a_loss.append(a_loss)
            img = image.array_to_img(images_fake[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_sample' + str(step) + '.png'))
            img = image.array_to_img(images_train[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_sample' + str(step) + '.png'))

    def build128(self,data_path,model_path,save_dir,iterations=1000,batch_size=64,height=32,width=32,latent_dim=100,channels=3):

        """设置相关参数"""
        # 潜变量维度
        # latent_dim = 100
        # 输入像素维度
        # height = 128
        # width = 128
        # channels = 3

        # 搭建生成器网络（Model方式）
        generator_input = Input(shape=(latent_dim,))
        x = Dense(128 * 64 * 64)(generator_input)
        x = LeakyReLU()(x)
        x = Reshape((64, 64, 128))(x)

        # IN: 16*16*128  OUT: 16*16*256
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # IN: 16*16*256  OUT: 32*32*256
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        # 通过反卷积操作，把维度变得和图片一样大了，就别用反卷积了
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # 把通道变回来
        x = Conv2D(channels, 7, activation='tanh', padding='same')(x)

        generator = Model(generator_input, x)
        # generator.summary()

        # 搭建判别器，判别器就真的是一个卷积神经网络了
        discriminator_input = Input(shape=(height, width, channels))
        # IN:32*32*3   OUT: 30*30*128
        x = Conv2D(128, 3)(discriminator_input)
        x = LeakyReLU()(x)

        # IN: 30*30*128  OUT:14*14*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:14*14*128   OUT:6*6*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:6*6*128  OUT:2*2*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # 展开成512个神经元
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(1, activation='sigmoid')(x)

        discriminator = Model(discriminator_input, x)
        # discriminator.summary()

        discriminator_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0008,
                                                     clipvalue=1.0,
                                                     decay=1e-8)

        discriminator.compile(optimizer=discriminator_optimizer,
                              loss='binary_crossentropy')

        # 将上面两个连起来组成生成对抗网络
        # 将判别器参数设置为不可训练
        discriminator.trainable = False

        gan_input = Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        # 搭建对抗网络
        gan = Model(gan_input, gan_output)
        gan_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0004,
                                           clipvalue=1.0,
                                           decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

        ## 下面导入数据集进行训练，数据集选用的是Keras自带的CIFAR-10数据集
        import os
        from keras.preprocessing import image

        # 加载数据集（手写数字的数据集也可以这样加载）
        # (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
        # 指定青蛙图像（编号为6）
        # x_train = x_train[y_train.flatten() == 6]
        # Load hdf5 dataset

        import h5py
        train_dataset = h5py.File(data_path, 'r')
        train_set_x_orig = np.array(train_dataset['X_train'][:])  # your train set features
        train_set_y_orig = np.array(train_dataset['y_train'][:])  # your train set labels
        test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
        test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
        train_dataset.close()

        print(train_set_x_orig.shape)
        print(train_set_y_orig.shape)

        print(train_set_x_orig.max())
        print(train_set_x_orig.min())

        print(test_set_x_orig.shape)
        print(test_set_y_orig.shape)
        x_train = train_set_x_orig

        print("data size:", x_train.shape)
        x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

        print("data size(adjusted):", x_train.shape)

        # 下面开始训练
        # iterations = 10000
        # batch_size = 16
        # save_dir = 'image128'

        start = 0
        for step in range(iterations):

            # 先通过噪声生成64张伪造图片
            noise = np.random.normal(size=(batch_size, latent_dim))
            images_fake = generator.predict(noise)

            # 从真实图片中抽64张
            end = start + batch_size
            images_train = x_train[start:end]

            # 两者混合作为训练集，并打上标签
            x = np.concatenate([images_fake, images_train])
            y = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])

            # 向标签中添加噪声
            y += 0.05 * np.random.random(y.shape)

            # 训练判别器
            d_loss = discriminator.train_on_batch(x, y)

            # 训练生成对抗网络
            noise = np.random.normal(size=(batch_size, latent_dim))
            labels = np.ones((batch_size, 1))

            # 通过gan模型来训练生成器模型，冻结判别器模型权重
            a_loss = gan.train_on_batch(noise, labels)

            start += batch_size
            if start > len(x_train) - batch_size:
                start = 0

            # 每500步绘图并保存
            if step % 100 == 0:
                generator.save(model_path)
            # if step % 10 == 0:
            print(f"Iteration={step + 1}\tDLoss={d_loss}\tALoss={a_loss}")
            # print('discriminator loss:', d_loss)
            # print('adversarial loss:', a_loss)
            self.d_loss.append(d_loss)
            self.a_loss.append(a_loss)
            img = image.array_to_img(images_fake[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_sample' + str(step) + '.png'))
            img = image.array_to_img(images_train[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_sample' + str(step) + '.png'))

    def build_gray(self,data_path,model_path,save_dir,iterations=1000,batch_size=64,height=32,width=32,latent_dim=100,channels=1):
        print("Building model...")
        """设置相关参数"""
        # 潜变量维度
        # latent_dim = 100
        # 输入像素维度
        # height = 32
        # width = 32
        # channels = 3

        # 搭建生成器网络（Model方式）
        generator_input = Input(shape=(latent_dim,))
        x = Dense(128 * 16 * 16)(generator_input)
        x = LeakyReLU()(x)
        x = Reshape((16, 16, 128))(x)

        # IN: 16*16*128  OUT: 16*16*256
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # IN: 16*16*256  OUT: 32*32*256
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        # 通过反卷积操作，把维度变得和图片一样大了，就别用反卷积了
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # 把通道变回来
        x = Conv2D(channels, 7, activation='tanh', padding='same')(x)

        generator = Model(generator_input, x)
        # generator.summary()

        # 搭建判别器，判别器就真的是一个卷积神经网络了
        discriminator_input = Input(shape=(height, width, channels))
        # IN:32*32*3   OUT: 30*30*128
        x = Conv2D(128, channels)(discriminator_input)
        x = LeakyReLU()(x)

        # IN: 30*30*128  OUT:14*14*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:14*14*128   OUT:6*6*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:6*6*128  OUT:2*2*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # 展开成512个神经元
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(1, activation='sigmoid')(x)

        discriminator = Model(discriminator_input, x)
        # discriminator.summary()

        discriminator_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0008,
                                                                clipvalue=1.0,
                                                                decay=1e-8)

        discriminator.compile(optimizer=discriminator_optimizer,
                              loss='binary_crossentropy')

        # 将上面两个连起来组成生成对抗网络
        # 将判别器参数设置为不可训练
        discriminator.trainable = False

        gan_input = Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        # 搭建对抗网络
        gan = Model(gan_input, gan_output)
        gan_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0004,
                                                      clipvalue=1.0,
                                                      decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

        ## 下面导入数据集进行训练，数据集选用的是Keras自带的CIFAR-10数据集


        # 加载数据集（手写数字的数据集也可以这样加载）
        # (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
        # 指定青蛙图像（编号为6）
        # x_train = x_train[y_train.flatten() == 6]
        # Load hdf5 dataset

        train_dataset = h5py.File(data_path, 'r')
        train_set_x_orig = np.array(train_dataset['X_train'][:])  # your train set features
        train_set_y_orig = np.array(train_dataset['y_train'][:])  # your train set labels
        test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
        test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
        train_dataset.close()

        print(train_set_x_orig.shape)
        print(train_set_y_orig.shape)

        print(train_set_x_orig.max())
        print(train_set_x_orig.min())

        print(test_set_x_orig.shape)
        print(test_set_y_orig.shape)
        x_train = train_set_x_orig

        print("data size:", x_train.shape)
        x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

        print("data size(adjusted):", x_train.shape)

        # 下面开始训练
        # iterations = 10000
        # batch_size = 64
        # save_dir = 'outputs/images32'


        start = 0
        for step in range(iterations):

            # 先通过噪声生成64张伪造图片
            noise = np.random.normal(size=(batch_size, latent_dim))
            images_fake = generator.predict(noise)

            # 从真实图片中抽64张
            end = start + batch_size
            images_train = x_train[start:end]

            # 两者混合作为训练集，并打上标签
            x = np.concatenate([images_fake, images_train])
            y = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])

            # 向标签中添加噪声
            y += 0.05 * np.random.random(y.shape)

            # 训练判别器
            d_loss = discriminator.train_on_batch(x, y)

            # 训练生成对抗网络
            noise = np.random.normal(size=(batch_size, latent_dim))
            labels = np.ones((batch_size, 1))

            # 通过gan模型来训练生成器模型，冻结判别器模型权重
            a_loss = gan.train_on_batch(noise, labels)

            start += batch_size
            if start > len(x_train) - batch_size:
                start = 0

            # 每500步绘图并保存
            if step % 100 == 0:
                generator.save(model_path)
            # if step % 10 == 0:
            print(f"Iteration={step+1}\tDLoss={d_loss}\tALoss={a_loss}")
            # print('discriminator loss:', d_loss)
            # print('adversarial loss:', a_loss)
            self.d_loss.append(d_loss)
            self.a_loss.append(a_loss)
            img = image.array_to_img(images_fake[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_sample' + str(step) + '.png'))
            img = image.array_to_img(images_train[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_sample' + str(step) + '.png'))


    def build(self,data_path,model_path,save_dir,iterations=1000,batch_size=64,height=32,width=32,latent_dim=100,channels=3):
        print("Building model...")
        """设置相关参数"""
        # 潜变量维度
        # latent_dim = 100
        # 输入像素维度
        # height = 32
        # width = 32
        # channels = 3

        # 搭建生成器网络（Model方式）
        generator_input = Input(shape=(latent_dim,))
        x = Dense(128 * 16 * 16)(generator_input)
        x = LeakyReLU()(x)
        x = Reshape((16, 16, 128))(x)

        # IN: 16*16*128  OUT: 16*16*256
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # IN: 16*16*256  OUT: 32*32*256
        x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
        x = LeakyReLU()(x)

        # 通过反卷积操作，把维度变得和图片一样大了，就别用反卷积了
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)
        x = Conv2D(256, 5, padding='same')(x)
        x = LeakyReLU()(x)

        # 把通道变回来
        x = Conv2D(channels, 7, activation='tanh', padding='same')(x)

        generator = Model(generator_input, x)
        # generator.summary()

        # 搭建判别器，判别器就真的是一个卷积神经网络了
        discriminator_input = Input(shape=(height, width, channels))
        # IN:32*32*3   OUT: 30*30*128
        x = Conv2D(128, 3)(discriminator_input)
        x = LeakyReLU()(x)

        # IN: 30*30*128  OUT:14*14*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:14*14*128   OUT:6*6*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # IN:6*6*128  OUT:2*2*128
        x = Conv2D(128, 4, strides=2)(x)
        x = LeakyReLU()(x)

        # 展开成512个神经元
        x = Flatten()(x)
        x = Dropout(0.4)(x)
        x = Dense(1, activation='sigmoid')(x)

        discriminator = Model(discriminator_input, x)
        # discriminator.summary()

        discriminator_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0008,
                                                                clipvalue=1.0,
                                                                decay=1e-8)

        discriminator.compile(optimizer=discriminator_optimizer,
                              loss='binary_crossentropy')

        # 将上面两个连起来组成生成对抗网络
        # 将判别器参数设置为不可训练
        discriminator.trainable = False

        gan_input = Input(shape=(latent_dim,))
        gan_output = discriminator(generator(gan_input))
        # 搭建对抗网络
        gan = Model(gan_input, gan_output)
        gan_optimizer = optimizers.rmsprop_v2.RMSprop(lr=0.0004,
                                                      clipvalue=1.0,
                                                      decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

        ## 下面导入数据集进行训练，数据集选用的是Keras自带的CIFAR-10数据集


        # 加载数据集（手写数字的数据集也可以这样加载）
        # (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
        # 指定青蛙图像（编号为6）
        # x_train = x_train[y_train.flatten() == 6]
        # Load hdf5 dataset

        train_dataset = h5py.File(data_path, 'r')
        train_set_x_orig = np.array(train_dataset['X_train'][:])  # your train set features
        train_set_y_orig = np.array(train_dataset['y_train'][:])  # your train set labels
        test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
        test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
        train_dataset.close()

        print(train_set_x_orig.shape)
        print(train_set_y_orig.shape)

        print(train_set_x_orig.max())
        print(train_set_x_orig.min())

        print(test_set_x_orig.shape)
        print(test_set_y_orig.shape)
        x_train = train_set_x_orig

        print("data size:", x_train.shape)
        x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

        print("data size(adjusted):", x_train.shape)

        # 下面开始训练
        # iterations = 10000
        # batch_size = 64
        # save_dir = 'outputs/images32'


        start = 0
        for step in range(iterations):

            # 先通过噪声生成64张伪造图片
            noise = np.random.normal(size=(batch_size, latent_dim))
            images_fake = generator.predict(noise)

            # 从真实图片中抽64张
            end = start + batch_size
            images_train = x_train[start:end]

            # 两者混合作为训练集，并打上标签
            x = np.concatenate([images_fake, images_train])
            y = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])

            # 向标签中添加噪声
            y += 0.05 * np.random.random(y.shape)

            # 训练判别器
            d_loss = discriminator.train_on_batch(x, y)

            # 训练生成对抗网络
            noise = np.random.normal(size=(batch_size, latent_dim))
            labels = np.ones((batch_size, 1))

            # 通过gan模型来训练生成器模型，冻结判别器模型权重
            a_loss = gan.train_on_batch(noise, labels)

            start += batch_size
            if start > len(x_train) - batch_size:
                start = 0

            # 每500步绘图并保存
            if step % 100 == 0:
                generator.save(model_path)
            # if step % 10 == 0:
            print(f"Iteration={step+1}\tDLoss={d_loss}\tALoss={a_loss}")
            # print('discriminator loss:', d_loss)
            # print('adversarial loss:', a_loss)
            self.d_loss.append(d_loss)
            self.a_loss.append(a_loss)
            img = image.array_to_img(images_fake[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'generated_sample' + str(step) + '.png'))
            img = image.array_to_img(images_train[0] * 255., scale=False)
            img.save(os.path.join(save_dir, 'real_sample' + str(step) + '.png'))

    # example of loading the generator model and generating images

    # generate points in latent space as input for the generator
    def generate_latent_points(self,latent_dim, n_samples):
        # generate points in the latent space
        x_input = randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, latent_dim)
        return x_input

    # create and save a plot of generated images (reversed grayscale)
    def show_plot(self,examples, n):
        # plot images
        for i in range(n * n):
            # define subplot
            pyplot.subplot(n, n, 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            # pyplot.imshow((examples[i] * 255).astype(np.uint8), cmap='jet')
            pyplot.imshow((examples[i] * 255).astype(np.uint8),cmap='gray')
        pyplot.show()


    def generate(self,n_samples=10,latent_dim=100,show=True):
        print("Generating sample from model")
        model_path=self.root_path+"/model"
        # load model
        model = load_model(model_path)
        model.summary()
        # generate images
        latent_points = self.generate_latent_points(latent_dim, n_samples*n_samples)
        # noise = np.random.normal(size=(64, 100))
        # generate images
        X = model.predict(latent_points)
        # plot the result
        if show:
            self.show_plot(X, n_samples)
        return  X

    def build_model(self,iterations=1000,train_dir="",use_gray=False):
        if train_dir=="":
            train_dir = f"{self.root_path}/data"
        out_path = f"{self.root_path}/data.h5"
        model_path = f"{self.root_path}/model"
        generated_path = f"{self.root_path}/generated"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        if not os.path.exists(generated_path):
            os.mkdir(generated_path)

        if not use_gray:
            self.create_dataset(train_dir=train_dir, out_path=out_path,use_gray=False)
        else:
            self.create_dataset_gray(train_dir=train_dir, out_path=out_path)

        if self.dim==32:
            if use_gray:
                self.build_gray(data_path=out_path, model_path=model_path, save_dir=generated_path, iterations=iterations,
                           width=self.dim, height=self.dim,channels=1)
            else:
                self.build(data_path=out_path, model_path=model_path, save_dir=generated_path,iterations=iterations,width=self.dim,height=self.dim)
        elif self.dim==64:
            if use_gray:
                self.build64(data_path=out_path, model_path=model_path, save_dir=generated_path, iterations=iterations,
                           width=self.dim, height=self.dim,channels=1)
            else:
                self.build64(data_path=out_path, model_path=model_path, save_dir=generated_path, iterations=iterations,
                       width=self.dim, height=self.dim)
        elif self.dim==128:
            self.build128(data_path=out_path, model_path=model_path, save_dir=generated_path, iterations=iterations,
                         width=self.dim, height=self.dim)
        else:
            raise Exception("not found dim!")
        # self.generate(model_path=model_path)

    def export_performance(self,export_file=""):
        if export_file=="":
            export_folder=self.root_path+"/performance"
            if not os.path.exists(export_folder):
                os.mkdir(export_folder)
            export_file=f"{export_folder}/loss.csv"
        f_out=open(export_file,"w",encoding='utf-8')
        f_out.write("Iteration\tLoss(A)\tLoss(D)\n")
        for idx,a in enumerate(self.a_loss):
            f_out.write(f"{idx}\t{self.a_loss[idx]}\t{self.d_loss[idx]}\n")
        f_out.close()







