from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

class PSNRSSIMCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        psnr = calcPSNR(self.y_val, y_pred)
        ssim = calcSSIM(self.y_val, y_pred, multichannel=True)
        print(f"\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.2f}")

#function for the colour transform from RGB to YCbCr space
def RGBToYCbCr_transform(image):
    #read the image planes into RGB values
    R, G, B = image[:,:,2], image[:,:,1], image[:,:,0]
    #apply conversion equations from RGB to YCbCr
    Y = np.clip(np.round(16 + (65.738*R/255)+(129.057*G/255)+(25.064*B/255)), 0, 255).astype(np.uint8)
    Cb = np.clip(np.round(128 - (37.945*R/255) - (74.494*G/255) + (112.439*B/255)), 0, 255).astype(np.uint8)
    Cr = np.clip(np.round(128 + (112.439*R/255) - (94.154*G/255) - (18.285*B/255)), 0, 255).astype(np.uint8)
    #stack 8-bit arrays in 3rd dimension (plane-wise)
    return np.stack([Y, Cb, Cr], axis=2).astype(np.uint8)

#function for the colour transform from YCbCr to RGB
def YCbCrtoRGB_transform(image):
    #read the image planes into YCbCr values
    Y, Cb, Cr = image[:,:,0], image[:,:,1], image[:,:,2]
    #apply conversion equations from YCbCr to RGB
    R = np.clip(np.round((298.082*Y/255) + (408.583*Cr/255) - 222.921), 0, 255).astype(np.uint8)
    G = np.clip(np.round((298.082*Y/255) - (100.291*Cb/255) - (208.120*Cr/255) + 135.576), 0, 255).astype(np.uint8)
    B = np.clip(np.round((298.082*Y/255) + (516.412* Cb/255) - 276.836), 0, 255).astype(np.uint8)
    #stack 8-bit arrays in 3rd dimension (plane-wise)
    return np.stack([R, G, B], axis=2).astype(np.uint8)



from tensorflow.keras.layers import BatchNormalization

def get_model(factor):
    input = Input(shape=(None, None, 1))
    x = Conv2D(32, 3, activation = 'relu', padding = 'same')(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(factor)(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, 3, activation = 'relu', padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, 3, activation = None, padding = 'same')(x)
    x = Activation('tanh')(x)
    x = x * 127.5 + 127.5

    model = Model([input],x)
    model.summary()
    return model



def get_data(factor, factorOther):
    Y_x = []
    Y_y = []
    Cb_x = []
    Cb_y = []
    Cr_x = []
    Cr_y = []
    # for img_dir in tqdm(glob('C://Users//User//Downloads//DIV2K_train_HR//DIV2K_train_HR//*.png')):
    for img_dir in tqdm(glob('D://Downloads//DIV2K_train_HR//DIV2K_train_HR//*.png')):
        img = cv2.imread(img_dir)
        img_ycrcb = RGBToYCbCr_transform(img)
        y_channel = img_ycrcb[:,:,0]
        cb_channel = img_ycrcb[:,:,1]
        cr_channel = img_ycrcb[:,:,2]


        y_out = cv2.resize(y_channel, (128,128), interpolation=cv2.INTER_AREA)
        y_in = cv2.resize(y_out, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_AREA)
        Y_x.append(y_in)
        Y_y.append(y_out)
        cb_out = cv2.resize(cb_channel, (128, 128), interpolation=cv2.INTER_AREA)
        cb_in = cv2.resize(cb_out, None, fx=1/factorOther, fy=1/factorOther, interpolation=cv2.INTER_AREA)
        Cb_x.append(cb_in)
        Cb_y.append(cb_out)
        cr_out = cv2.resize(cr_channel, (128, 128), interpolation=cv2.INTER_AREA)
        cr_in = cv2.resize(cr_out, None, fx=1/factorOther, fy=1/factorOther, interpolation=cv2.INTER_AREA)
        Cr_x.append(cr_in)
        Cr_y.append(cr_out)

    Y_x = np.array(Y_x)
    Y_y = np.array(Y_y)
    Cb_x = np.array(Cb_x)
    Cb_y = np.array(Cb_y)
    Cr_x = np.array(Cr_x)
    Cr_y = np.array(Cr_y)
    return Y_x,Y_y,Cb_x,Cb_y,Cr_x,Cr_y

def doModel(modelInput, modelName, x, y):
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 42)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) #can be tuned as a hyperparameter
    loss = 'mse' # can be other losses
    modelInput.compile(loss=loss, optimizer=optimizer)

    save_model_callback = tf.keras.callbacks.ModelCheckpoint(
        'model/'+modelName+'.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_freq = 'epoch')
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq = 0, write_graph = True, write_images = True)

    batch_size = 4
    epochs = 10 #100 maybe!
    # can get data loader as input
    # psnr_ssim_callback = PSNRSSIMCallback(X_val, y_val)
    modelInput.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size = batch_size, epochs = epochs, validation_split=0.1, callbacks=[tbCallBack, save_model_callback])


#function to compute the PSNR value
def calcPSNR(before_image, after_image):
    #calculate the mean squared value of the difference (error)
    mse = np.mean((before_image - after_image) ** 2)
    #find the Peak-Signal-to-Noise-Ratio
    psnr = 10 * np.log10(255**2 / mse)
    return psnr

def calcSSIM(img1, img2):
    # Convert the images to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Compute the mean and variance of the images
    mean1, var1 = cv2.meanStdDev(img1)
    mean2, var2 = cv2.meanStdDev(img2)

    # Compute the covariance and cross-covariance of the images
    cov = np.cov(img1.ravel(), img2.ravel())[0][1]
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    # Compute the SSIM score
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 ** 2 + var2 ** 2 + c2))
    return ssim

#function to downsample input by a scale factor
def downsample(channel, scale_factor):
    #extract height and width values for the image
    height, width = len(channel),len(channel[0])

    #initialize output colour spaces by calculated height and width after downsample
    downsampled_channel = np.zeros((height//scale_factor, width//scale_factor))

    #loop to end of dimensions, skip by every factor value
    for i in range(0, height, scale_factor):
        for j in range(0, width, scale_factor):
            if (j+scale_factor) <= width:
                #sliding window based on the scale factor for each plane
                window_channel = channel[i:i+scale_factor, j:j+scale_factor]
                #average the values within each area
                avg_channel = np.mean(window_channel)
                #plot the 8-bit values within a scaled down area
                downsampled_channel[i//scale_factor, j//scale_factor] = avg_channel.astype(np.uint8)
    #stack 8-bit arrays in 3rd dimension (plane-wise)
    return downsampled_channel

factor = 2
factorOther = 4
modelY = get_model(factor)
modelCb = get_model(factorOther)
modelCr = get_model(factorOther)
Y_x,Y_y,Cb_x,Cb_y,Cr_x,Cr_y = get_data(factor,factorOther)
doModel(modelY,"modelY2", Y_x, Y_y)
doModel(modelCb,"modelCb2", Cb_x, Cb_y)
doModel(modelCr,"modelCr2", Cr_x, Cr_y)

# # plt.subplot(211)
# # plt.imshow(x[0], cmap = 'gray')
# # plt.subplot(212)
# # plt.imshow(y[0], cmap = 'gray')
# # plt.show()

modelY = load_model('model/modelY2.h5')
modelCb = load_model('model/modelCb2.h5')
modelCr = load_model('model/modelCr2.h5')
img = cv2.imread('LebronJames.jpg')
# img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
img_ycrcb = RGBToYCbCr_transform(img)
y_channel = img_ycrcb[:,:,0]
y_channel = downsample(y_channel,factor)
y_in = cv2.resize(y_channel, (y_channel.shape[1],y_channel.shape[0]), interpolation=cv2.INTER_AREA)
y = np.expand_dims(y_in, axis=0)
y_upsampled = modelY.predict(y)

cb_channel = img_ycrcb[:,:,1]
cb_channel = downsample(cb_channel,factorOther)
cb_in = cv2.resize(cb_channel, (cb_channel.shape[1],cb_channel.shape[0]), interpolation=cv2.INTER_AREA)
cb = np.expand_dims(cb_in, axis=0)
cb_upsampled = modelCb.predict(cb)

cr_channel = img_ycrcb[:,:,2]
cr_channel = downsample(cr_channel,factorOther)
cr_in = cv2.resize(cr_channel, (cr_channel.shape[1],cr_channel.shape[0]), interpolation=cv2.INTER_AREA)
cr = np.expand_dims(cr_in, axis=0)
cr_upsampled = modelCr.predict(cr)

print(np.squeeze(y_upsampled).shape)
print(np.squeeze(cb_upsampled).shape)
print(np.squeeze(cr_upsampled).shape)
result_ycrcb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
result_ycrcb[:,:,0] = cv2.resize(np.squeeze(y_upsampled),(img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
result_ycrcb[:,:,1] = cv2.resize(np.squeeze(cb_upsampled),(img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
result_ycrcb[:,:,2] = cv2.resize(np.squeeze(cr_upsampled),(img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
result = YCbCrtoRGB_transform(result_ycrcb)

print(img.shape)
print(result.shape)
# plt.subplot(211)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.subplot(212)
plt.imshow(result)
plt.show()

print("PSNR: "+str(calcPSNR(img,result)))
print("SSIM: "+str(calcSSIM(img,result)))