import sys
import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras import optimizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.models import load_model
from matplotlib import pyplot
from numpy import asarray
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import h5py
import cv2
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class CGanNode:

	def __init__(self,name,root_path=""):
		self.root_path=f"{root_path}/{name}"
		if not os.path.exists(self.root_path):
			os.mkdir(self.root_path)
		self.model_path=f"{self.root_path}/{name}.h5"
		self.data_path=f"{self.root_path}/data.h5"
		self.metrics=[]


	def get_files(self,file_dir):

		labels=os.listdir(file_dir)

		files_set = []
		lbs_set = []
		for idx, label in enumerate(labels):
			files = []
			lbs = []
			for file in os.listdir(file_dir + '/' + label):
				files.append(file_dir + '/' + label + '/' + file)
				lbs.append(idx)  # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
			files_set.append(files)
			lbs_set.append(lbs)

		files_tuple = ()
		lbs_tuple = ()
		for idx, label in enumerate(labels):
			files_tuple = files_tuple + tuple(files_set[idx])
			lbs_tuple = lbs_tuple + tuple(lbs_set[idx])

		# 把cat和dog合起来组成一个list（img和lab）
		image_list = np.hstack(files_tuple)
		label_list = np.hstack(lbs_tuple)

		# 利用shuffle打乱顺序
		temp = np.array([image_list, label_list])
		temp = temp.transpose()
		np.random.shuffle(temp)

		# 从打乱的temp中再取出list（img和lab）
		image_list = list(temp[:, 0])
		label_list = list(temp[:, 1])
		label_list = [int(i) for i in label_list]

		return image_list, label_list

	def create_datasets(self,train_dir=""):
		if train_dir=="":
			train_dir=self.data_path+"/data"
		# train_dir = 'datasets/Medical_MNIST'
		image_list, label_list = self.get_files(train_dir)

		# print(len(image_list))
		# print(len(label_list))

		# 450为数据长度的20%
		percent20 = int(len(image_list) * 0.2)
		dim = 28
		Train_image = np.random.rand(len(image_list) - percent20, dim, dim).astype('float32')
		Train_label = np.random.rand(len(image_list) - percent20, 1).astype('float32')

		Test_image = np.random.rand(percent20, dim, dim).astype('float32')
		Test_label = np.random.rand(percent20, 1).astype('float32')

		for i in range(len(image_list) - percent20):
			img = plt.imread(image_list[i])
			# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
			Train_image[i] = np.array(shrink_img)
			Train_label[i] = np.array(label_list[i])

		for i in range(len(image_list) - percent20, len(image_list)):
			img = plt.imread(image_list[i])
			# img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			shrink_img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_AREA)
			Test_image[i + percent20 - len(image_list)] = np.array(shrink_img)
			Test_label[i + percent20 - len(image_list)] = np.array(label_list[i])

		# Create a new file
		f = h5py.File(self.data_path, 'w')
		f.create_dataset('X_train', data=Train_image)
		f.create_dataset('y_train', data=Train_label)
		f.create_dataset('X_test', data=Test_image)
		f.create_dataset('y_test', data=Test_label)
		f.close()

	# define the standalone discriminator model
	def define_discriminator(self,in_shape=(28,28,1), n_classes=10):
		# label input
		in_label = Input(shape=(1,))
		# embedding for categorical input
		li = Embedding(n_classes, 50)(in_label)
		# scale up to image dimensions with linear activation
		n_nodes = in_shape[0] * in_shape[1]
		li = Dense(n_nodes)(li)
		# reshape to additional channel
		li = Reshape((in_shape[0], in_shape[1], 1))(li)
		# image input
		in_image = Input(shape=in_shape)
		# concat label as a channel
		merge = Concatenate()([in_image, li])
		# downsample
		fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
		fe = LeakyReLU(alpha=0.2)(fe)
		# downsample
		fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
		fe = LeakyReLU(alpha=0.2)(fe)
		# flatten feature maps
		fe = Flatten()(fe)
		# dropout
		fe = Dropout(0.4)(fe)
		# output
		out_layer = Dense(1, activation='sigmoid')(fe)
		# define model
		model = Model([in_image, in_label], out_layer)
		# compile model
		opt = optimizers.adam_v2.Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
		return model

	# define the standalone generator model
	def define_generator(self,latent_dim, n_classes=10):
		# label input
		in_label = Input(shape=(1,))
		# embedding for categorical input
		li = Embedding(n_classes, 50)(in_label)
		# linear multiplication
		n_nodes = 7 * 7
		li = Dense(n_nodes)(li)
		# reshape to additional channel
		li = Reshape((7, 7, 1))(li)
		# image generator input
		in_lat = Input(shape=(latent_dim,))
		# foundation for 7x7 image
		n_nodes = 128 * 7 * 7
		gen = Dense(n_nodes)(in_lat)
		gen = LeakyReLU(alpha=0.2)(gen)
		gen = Reshape((7, 7, 128))(gen)
		# merge image gen and label input
		merge = Concatenate()([gen, li])
		# upsample to 14x14
		gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
		gen = LeakyReLU(alpha=0.2)(gen)
		# upsample to 28x28
		gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
		gen = LeakyReLU(alpha=0.2)(gen)
		# output
		out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
		# define model
		model = Model([in_lat, in_label], out_layer)
		return model

	# define the combined generator and discriminator model, for updating the generator
	def define_gan(self,g_model, d_model):
		# make weights in the discriminator not trainable
		d_model.trainable = False
		# get noise and label inputs from generator model
		gen_noise, gen_label = g_model.input
		# get image output from the generator model
		gen_output = g_model.output
		# connect image output and label input from generator as inputs to discriminator
		gan_output = d_model([gen_output, gen_label])
		# define gan model as taking noise and label and outputting a classification
		model = Model([gen_noise, gen_label], gan_output)
		# compile model
		opt = optimizers.adam_v2.Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss='binary_crossentropy', optimizer=opt)
		return model

	# load fashion mnist images
	def load_real_samples(self):
		# load dataset
		(trainX, trainy), (_, _) = load_data()
		# expand to 3d, e.g. add channels
		X = expand_dims(trainX, axis=-1)
		# convert from ints to floats
		X = X.astype('float32')
		# scale from [0,255] to [-1,1]
		X = (X - 127.5) / 127.5
		return [X, trainy]

	def load_my_samples(self,data_path):
		# load dataset
		import h5py
		import numpy as np
		# (trainX, _), (_, _) = load_data()
		train_dataset = h5py.File(data_path, 'r')
		train_set_x_orig = np.array(train_dataset['X_train'][:])  # your train set features
		train_set_y_orig = np.array(train_dataset['y_train'][:])  # your train set labels
		test_set_x_orig = np.array(train_dataset['X_test'][:])  # your train set features
		test_set_y_orig = np.array(train_dataset['y_test'][:])  # your train set labels
		train_dataset.close()

		# print(train_set_x_orig.shape)
		#print(train_set_y_orig.shape)

		#print(train_set_x_orig.max())
		#print(train_set_x_orig.min())

		#print(test_set_x_orig.shape)
		#print(test_set_y_orig.shape)
		X=train_set_x_orig
		# expand to 3d, e.g. add channels
		X = expand_dims(train_set_x_orig, axis=-1)
		# print("expand dims")
		# print(X.shape)
		# convert from ints to floats
		X = X.astype('float32')
		# scale from [0,255] to [-1,1]
		X = (X - 127.5) / 127.5
		return [X, train_set_y_orig]

	# # select real samples
	def generate_real_samples(self,dataset, n_samples):
		# split into images and labels
		images, labels = dataset
		# choose random instances
		ix = randint(0, images.shape[0], n_samples)
		# select images and labels
		X, labels = images[ix], labels[ix]
		# generate class labels
		y = ones((n_samples, 1))
		return [X, labels], y

	# generate points in latent space as input for the generator
	def generate_latent_points(self,latent_dim, n_samples, n_classes=10):
		# generate points in the latent space
		x_input = randn(latent_dim * n_samples)
		# reshape into a batch of inputs for the network
		z_input = x_input.reshape(n_samples, latent_dim)
		# generate labels
		labels = randint(0, n_classes, n_samples)
		return [z_input, labels]

	# use the generator to generate n fake examples, with class labels
	def generate_fake_samples(self,generator, latent_dim, n_samples):
		# generate points in latent space
		z_input, labels_input = self.generate_latent_points(latent_dim, n_samples)
		# predict outputs
		images = generator.predict([z_input, labels_input])
		# create class labels
		y = zeros((n_samples, 1))
		return [images, labels_input], y

	# train the generator and discriminator
	def train(self,g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128,model_path=""):
		bat_per_epo = int(dataset[0].shape[0] / n_batch)
		half_batch = int(n_batch / 2)
		# manually enumerate epochs
		for i in range(n_epochs):
			# enumerate batches over the training set
			for j in range(bat_per_epo):
				# get randomly selected 'real' samples
				[X_real, labels_real], y_real = self.generate_real_samples(dataset, half_batch)
				# update discriminator model weights
				d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
				# generate 'fake' examples
				[X_fake, labels], y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
				# update discriminator model weights
				d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
				# prepare points in latent space as input for the generator
				[z_input, labels_input] = self.generate_latent_points(latent_dim, n_batch)
				# create inverted labels for the fake samples
				y_gan = ones((n_batch, 1))
				# update the generator via the discriminator's error
				g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
				# summarize loss on this batch
				print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
					(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
				self.metrics.append([str(i+1),str(j+1),str(bat_per_epo),str(d_loss1),str(d_loss2),str(g_loss)])

		# save the generator model
		g_model.save(model_path)

	def export_performance(self,export_file=""):
		if export_file=="":
			export_file=self.root_path+"/performance.csv"
		f_out=open(export_file,"w",encoding='utf-8')
		for metric in self.metrics:
			line="\t".join(metric)
			f_out.write(line+"\n")
		f_out.close()

	def build(self,n_epochs=10):
		# size of the latent space
		latent_dim = 100
		# create the discriminator
		d_model = self.define_discriminator()
		# create the generator
		g_model = self.define_generator(latent_dim)
		# create the gan
		gan_model = self.define_gan(g_model, d_model)
		# load image data
		dataset = self.load_my_samples(data_path=self.data_path)
		# train model
		self.train(g_model, d_model, gan_model, dataset, latent_dim,model_path=self.model_path,n_epochs=n_epochs)

	# create and save a plot of generated images
	def save_plot(self,examples, n):
		# plot images
		for i in range(n * n):
			# define subplot
			pyplot.subplot(n, n, 1 + i)
			# turn off axis
			pyplot.axis('off')
			# plot raw pixel data
			pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
		pyplot.show()

	def generate(self,n_samples=10):
		# load model
		model = load_model(self.model_path)
		# generate images
		latent_points, labels = self.generate_latent_points(100, 100)
		# specify labels
		labels = asarray([x for _ in range(n_samples) for x in range(10)])
		# generate images
		X = model.predict([latent_points, labels])
		# scale from [-1,1] to [0,1]
		X = (X + 1) / 2.0
		# plot the result
		self.save_plot(X, n_samples)
