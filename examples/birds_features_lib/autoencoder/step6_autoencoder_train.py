import matplotlib
matplotlib.use("Agg")
from mmkfeatures.image.conv_autoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from mmkfeatures.fusion.mm_features_lib import MMFeaturesLib
import numpy as np
import cv2
from tqdm import tqdm

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 20
INIT_LR = 1e-3
BS = 32
img_size=(64,64)
channel=3
split=0.8

#+++++++++++start+++++++++++++++

mmf_file=f"../datasets/birds_raw.mmf"
feature_lib=MMFeaturesLib(file_path=mmf_file)
data=feature_lib.get_data()

trainX=[]
testX=[]

list_cids=list(data.keys())
total_num=len(list_cids)
train_num=int(total_num*split)
test_num=total_num-train_num
list_cids_train=list_cids[:train_num]
list_cids_test=list_cids[train_num:]

for cid in tqdm(list_cids_train):
	item=data[cid]
	imgs=item["objects"]
	for img_id in imgs:
		# print(image)
		img=imgs[img_id][()]
		resized_img = cv2.resize(img, dsize=img_size)
		trainX.append(resized_img)

for cid in tqdm(list_cids_test):
	item=data[cid]
	imgs=item["objects"]
	for img_id in imgs:
		# print(image)
		img=imgs[img_id][()]
		resized_img=cv2.resize(img,dsize=img_size)
		testX.append(resized_img)

trainX=np.array(trainX)
testX=np.array(testX)

def visualize_predictions(decoded, gt, samples=10):
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

# configure
_model="output/autoencoder.h5"
_vis="output/recon_vis.png"
_plot="output/plot.png"


# load the MNIST dataset
print("[INFO] loading MNIST dataset...")
# ((trainX, _), (testX, _)) = mnist.load_data()


# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
# trainX = np.expand_dims(trainX, axis=-1)
# testX = np.expand_dims(testX, axis=-1)
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
# construct our convolutional autoencoder
print("[INFO] building autoencoder...")
autoencoder = ConvAutoencoder.build(img_size[0], img_size[1], channel)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
autoencoder.compile(loss="mse", optimizer=opt)
# train the convolutional autoencoder
H = autoencoder.fit(
	trainX, trainX,
	validation_data=(testX, testX),
	epochs=EPOCHS,
	batch_size=BS)

# use the convolutional autoencoder to make predictions on the
# testing images, construct the visualization, and then save it
# to disk
print("[INFO] making predictions...")
decoded = autoencoder.predict(testX)
vis = visualize_predictions(decoded, testX)
cv2.imwrite(_vis, vis)
# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(_plot)
# serialize the autoencoder model to disk
print("[INFO] saving autoencoder...")
autoencoder.save(_model, save_format="h5")

