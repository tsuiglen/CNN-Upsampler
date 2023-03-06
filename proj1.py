#import necessary libraries
import cv2 as cv
import numpy as np

#function for the colour transform from RGB to YCbCr space
def RGBToYCbCr_transform(image):
    #read the image planes into RGB values
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]
    #apply conversion equations from RGB to YCbCr
    Y = 16 + (65.738*R/256)+(129.057*G/256)+(25.064*B/256)
    Cb = 128 - (37.945*R/256) - (74.494*G/256) + (112.439*B/256)
    Cr = 128 + (112.439*R/256) - (94.154*G/256) - (18.285*B/256)
    #stack 8-bit arrays in 3rd dimension (plane-wise)
    return np.stack([Y, Cb, Cr], axis=2).astype(np.uint8)

#function for the colour transform from YCbCr to RGB
def YCbCrtoRGB_transform(image):
    #read the image planes into YCbCr values
    Y, Cb, Cr = image[:,:,0], image[:,:,1], image[:,:,2]
    #apply conversion equations from YCbCr to RGB
    R = (298.082*Y/256) + (408.583*Cr/256) - 222.921
    G = (298.082*Y/256) - (100.291*Cb/256) - (208.120*Cr/256) + 135.576
    B = (298.082*Y/256) + (516.412* Cb/256) - 276.836
    #stack 8-bit arrays in 3rd dimension (plane-wise)
    return np.stack([R, G, B], axis=2).astype(np.uint8)

#function to compute the PSNR value
def calcPSNR(before_image, after_image):
    #calculate the mean squared value of the difference (error)
    mse = np.mean((before_image - after_image) ** 2)
    #find the Peak-Signal-to-Noise-Ratio
    psnr = 10 * np.log10(255**2 / mse)
    return psnr

#function to downsample input by a scale factor
def downsample(image, scale_factor):
    #extract height and width values for the image
    height, width = len(image),len(image[0])
    #initialize colour spaces for the image
    Y = image[:,:,0]
    Cb = image[:, :, 1]
    Cr = image[:, :, 2]
    #initialize output colour spaces by calculated height and width after downsample
    downsampled_Y = np.zeros((height//scale_factor, width//scale_factor))
    downsampled_Cb = np.zeros((height//scale_factor, width//scale_factor))
    downsampled_Cr = np.zeros((height//scale_factor, width//scale_factor))
    #loop to end of dimensions, skip by every factor value
    for i in range(0, height, scale_factor):
        for j in range(0, width, scale_factor):
            #sliding window based on the scale factor for each plane
            window_Y = Y[i:i+scale_factor, j:j+scale_factor]
            window_Cb = Cb[i:i+scale_factor, j:j+scale_factor]
            window_Cr = Cr[i:i+scale_factor, j:j+scale_factor]
            #average the values within each area
            avg_Y = np.mean(window_Y)
            avg_Cb = np.mean(window_Cb)
            avg_Cr = np.mean(window_Cr)
            #plot the 8-bit values within a scaled down area
            downsampled_Y[i//scale_factor, j//scale_factor] = avg_Y.astype(np.uint8)
            downsampled_Cb[i//scale_factor, j//scale_factor] = avg_Cb.astype(np.uint8)
            downsampled_Cr[i//scale_factor, j//scale_factor] = avg_Cr.astype(np.uint8)
    #stack 8-bit arrays in 3rd dimension (plane-wise)
    return np.stack([downsampled_Y,downsampled_Cb,downsampled_Cr], axis=2).astype(np.uint8)

#function to perform bilinear interpolation for upsampling
def bilinear_interpolation(image, scale_factor):
    #extract height and width values for the image
    height, width = len(image),len(image[0])
    #initialize output height and width based on factor
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    #initialize output image based on upsamplign dimensions
    new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    #map out coordinates of each pixel by dividing by scale factor
    x = np.zeros(new_width)
    for i in range(new_width):
        x[i] = i / scale_factor
    # coordinates of new image pixels
    y = np.zeros(new_height)
    for i in range(new_height):
        y[i] = i / scale_factor
    #closest points to four points
    y_floor = np.floor(y).astype(np.int32)
    x_floor = np.floor(x).astype(np.int32)
    y_ceil = y_floor + 1
    y_ceil[y_ceil > height - 1] = height - 1
    x_ceil = x_floor + 1
    x_ceil[x_ceil > width - 1] = width - 1
    #compute difference values for equation
    dy = y - y_floor
    dx = x - x_floor
    #loop through newly initialized dimensions
    for i in range(new_height):
        for j in range(new_width):
            #define four pixel values to estimate upsample
            q11 = image[y_floor[i], x_floor[j]]
            q12 = image[y_floor[i], x_ceil[j]]
            q21 = image[y_ceil[i], x_floor[j]]
            q22 = image[y_ceil[i], x_ceil[j]]
            #apply bilinear interpolation formula
            new_image[i, j] = (
                (q11*(1 - dx[j])*(1 - dy[i])) + (q12*dx[j]*(1 - dy[i])) +  (q21*(1 - dx[j])*dy[i]) + (q22*dx[j]*dy[i])
            )
    #output uint8 upsampled image
    return new_image.astype(np.uint8)

#define downsampling and upsampling scale
scale = 2
#load input image
original_image = cv.imread('test image.png')
#transform from RGB to YCbCr space
ycbcr_image = RGBToYCbCr_transform(original_image)
cv.imwrite("ycbcrtest image.png", ycbcr_image)
#downsample image by defined factor
downsample_ycbcr_image = downsample(ycbcr_image, scale)
cv.imwrite("downsample_ycbcrtest image.png", downsample_ycbcr_image)
#upsample image by defined factor with bilinear interpolation
upsample_ycbcr_image = bilinear_interpolation(downsample_ycbcr_image,scale)
cv.imwrite("upsample_ycbcrtest image.png", upsample_ycbcr_image)
#transform from YCbCr to RGB space
compressed_image = YCbCrtoRGB_transform(upsample_ycbcr_image)
cv.imwrite("newtest image.png", compressed_image)
#calculate PSNR value
psnr = calcPSNR(original_image, compressed_image)
print('PSNR:',psnr)