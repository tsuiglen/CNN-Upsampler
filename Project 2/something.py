from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm

def get_model():
    input = Input(shape=(None, None, 1))
    x = Conv2D(32, 3, activation = 'relu', padding = 'same')(input)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(32, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(1, 3, activation = None, padding = 'same')(x)
    x = Activation('tanh')(x)
    x = x * 127.5 + 127.5

    model = Model([input],x)
    model.summary()
    return model


def get_data():
    x = []
    y = []
    for img_dir in tqdm(glob('C://Users//User//Downloads//DIV2K_train_HR//DIV2K_train_HR//*.png')):
        img = cv2.imread(img_dir)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = img_ycrcb[:,:,0]
        y_out = cv2.resize(y_channel, (128,128), interpolation=cv2.INTER_AREA)
        y_in = cv2.resize(y_out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        x.append(y_in)
        y.append(y_out)
    x = np.array(x)
    y = np.array(y)
    return x,y


model = get_model()
x,y = get_data()
print(x.shape,y.shape)
import matplotlib.pyplot as plt
# plt.subplot(211)
# plt.imshow(x[0], cmap = 'gray')
# plt.subplot(212)
# plt.imshow(y[0], cmap = 'gray')
# plt.show()

from sklearn.model_selection import train_test_split
import tensorflow as tf
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) #can be tuned as a hyperparameter
loss = 'mse' # can be other losses
model.compile(loss=loss, optimizer=optimizer)

save_model_callback = tf.keras.callbacks.ModelCheckpoint(
    'model/model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
    save_freq = 'epoch')
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq = 0, write_graph = True, write_images = True)

batch_size = 4
epochs = 10 #100 maybe!
# can get data loader as input
model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size = batch_size, epochs = epochs, validation_split=0.1, callbacks=[tbCallBack, save_model_callback])

model = load_model('model/model.h5')
img = cv2.imread('LebronJames.jpg')
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y_channel = img_ycrcb[:,:,0]
y_in = cv2.resize(y_channel, (256,256), interpolation=cv2.INTER_AREA)
y = np.expand_dims(y_in, axis=0)
y_upsampled = model.predict(y)
print(y_upsampled.shape)
plt.subplot(211)
plt.imshow(y[0], cmap = 'gray')
plt.subplot(212)
plt.imshow(y_upsampled[0], cmap = 'gray')
plt.show()
