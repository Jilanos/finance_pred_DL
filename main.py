import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import numpy as np 
import datetime
import os
import time
import sys
from data import loadData
from encoder import Encoder, Decoder, VisualAE


ignoreTimer = 150
n_points = 100
data = loadData(paire="BTCBUSD", sequenceLength=80*n_points, interval_str="{}m".format(5), numPartitions=4,trainProp = 0.6, validProp = 0.25, testProp  = 0.15, reload=False, ignoreTimer=ignoreTimer)


#on va maintenant découper ces données en segment de 80 points et mettre ça dans un array
data_cut = []
for i in range(n_points):
    data_cut.append(data[i*80:(i+1)*80])
# on va maintenant découper ces data en test train et valid en utilisant train_test_split
train, test = train_test_split(data_cut, test_size=0.15, shuffle=False)
train, valid = train_test_split(train, test_size=0.25, shuffle=False)

#Given a 1-d numeric time series 𝑆 = [𝑠0, · · · , 𝑠𝑇 ] with 𝑠𝑡 ∈ R, we convert 𝑆 into a 2-d image 𝑥 by plotting it out, with 𝑡 being the horizontal axis and 𝑠𝑡 being the vertical axis1. We standardize each converted image 𝑥 through following pre-processing steps. First, pixels in 𝑥 are scaled to [0, 1] and negated (i.e., 𝑥 = 1 − 𝑥/255) so that the pixels corresponding to the plotted time series signal are bright (values close to 1), whereas the rest of the background pixels become dark (values close to 0). Note that there can be multiple bright (non-zero) pixels in each column due to anti-aliasing while plotting the images. Upon normalizing each column in 𝑥 such that the pixel values in each column sum to 1, each column can be perceived as a discrete probability distribution (see Figure 6). Columns represent the independent variable time, while rows capture the dependent variable: pixel intensity. The value of the time series 𝑆 at time 𝑡 is now simply the pixel index 𝑟 (row) at that time (column) with the highest intensity. Predictions are made over normalized data. To preserve the ability to forecast in physical units, we utilize the span of the input raw data values to transform forecasts to the corresponding physical scales.
#%%
plt.close('all')
def make_image(df_array, size_image=80):
    min_val = np.min(df_array)
    max_val = np.max(df_array)
    normalized_array = (df_array - min_val) / (max_val - min_val)

    point_range = np.linspace(0, 1, size_image)[:, np.newaxis]
    distances = np.abs(point_range - normalized_array)

    below_threshold = distances <= 0.0125
    probabilities = np.sum(below_threshold, axis=1)

    probabilities = np.clip(probabilities, 0, 1).astype(float)
    probabilities /= np.sum(probabilities)

    array_image = probabilities[:, np.newaxis] * below_threshold.astype(float)

    return np.flipud(array_image)#np.flipud(array_image)

def series_to_img(serie):
    #on va convertir la série en image
    #on va normaliser la série
    epsilon = 1e-3
    n = len(serie)
    serie = np.array(serie)
    serie = (serie - serie.min() + epsilon)/(serie.max()-serie.min()+2*epsilon)
    #on veut que chaque colonne de l'image soit une distribution de probabilité
    # pour cela on va considérer le prix de fermeture comme le centre d'une gaussienne 
    # et on va considérer que la distribution de probabilité et on placera 3 sigma de chaque coté comme le high ou le low
    # ici le high est series[:,1] et le low est series[:,2]  
    #on va maintenant créer l'image
    img = np.zeros((n,n))
    for i in range(n):
        M = serie[i,0]
        ecart_high = (serie[i,1]-M)/1.0
        ecart_low = (M-serie[i,2])/1.0
        if ecart_high == 0:
            ecart_high = ecart_low
        elif ecart_low == 0:
            ecart_low = ecart_high
            
        for j in range(n):
            if float(j)/n > M:
                img[j,i] = np.exp(-0.5 * ((float(j)/n - M) / ecart_high) ** 2) / (ecart_high * np.sqrt(2 * np.pi))
            else:
                img[j,i] = np.exp(-0.5 * ((float(j)/n - M) / ecart_low) ** 2) / (ecart_low * np.sqrt(2 * np.pi))
        if list(img[:,i]) == list(np.zeros(np.shape(img[:,i]))):
            #alors on va mettre un unique pixel au plus proche du close
            img[int(serie[i,0]*n),i] = 1
    for i in range(n):
        img[:,i] = img[:,i]/np.sum(img[:,i])

    return np.flipud(img)


#As mentioned in Section 1, recent work has seen the extensive use of autoencoders in both the time series and computer vision domains. Following these, we extend the use of autoencoders to our image-to-image time series forecasting setting. We use a simplistic convolutional autoencoder to produce a visual forecast image with the continuation of an input time series image, by learning an undercomplete mapping 𝑔 ◦ 𝑓 , ˆ 𝑦 = 𝑔(𝑓 (𝑥)), ∀𝑥 ∈ 𝑋, where the encoder network 𝑓 (·) learns meaningful patterns and projects the input image 𝑥 into an embedding vector, and the decoder network 𝑔(·) reconstructs the forecast image from the embedding vector. We purposely do not use sequential information or LSTM cells as we wish to examine the benefits of framing the regression problem in an image setting.
#We call this method VisualAE, the architecture for which is shown in Figure 5. We used 2D convolutional layers with a kernel size of 5 × 5, stride 2, and padding 2. All layers are followed by ReLU activation and batch normalization. The encoder network consists of 3 convolutional layers which transform a 80 × 80 × 1 input image to 10 × 10 × 512, after which we obtain an embedding vector of length 512 using a fully connected layer. This process is then mirrored for the decoder network, resulting in a forecast image of dimension 80 × 80. We will explain the loss function for training in detail in the next section.


#on veut afficher les 10 premières images
for i in range(10):
    print(i)
    plt.figure(figsize=(20,10))
    #on veut afficher maintenant l'image en grayscale
    img1 = series_to_img(np.array(data_cut[i]))
    img2 = make_image(np.array(data_cut[i])[:,0])
    #on veut afficher les 2 image à coté
    plt.subplot(1,3,1)
    plt.imshow(1-img1, cmap='gray')
    plt.title('Mon image')
    plt.subplot(1,3,2)
    plt.imshow(1-img2, cmap='gray')
    plt.title('Ta version')
    plt.subplot(1,3,3)
    plt.plot(np.array(data_cut)[i,:,0],color='black',label='close')
    plt.plot(np.array(data_cut)[i,:,1],color='red',label='high')
    plt.plot(np.array(data_cut)[i,:,2],color='green',label='low')
    plt.legend(fontsize=15)
    plt.xlabel('time',fontsize=15)
    plt.ylabel('price',fontsize=15)



#As mentioned in Section 1, recent work has seen the extensive use of autoencoders in both the time series and computer vision domains. Following these, we extend the use of autoencoders to our image-to-image time series forecasting setting. We use a simplistic convolutional autoencoder to produce a visual forecast image with the continuation of an input time series image, by learning an undercomplete mapping 𝑔 ◦ 𝑓 , ˆ 𝑦 = 𝑔(𝑓 (𝑥)), ∀𝑥 ∈ 𝑋, where the encoder network 𝑓 (·) learns meaningful patterns and projects the input image 𝑥 into an embedding vector, and the decoder network 𝑔(·) reconstructs the forecast image from the embedding vector. We purposely do not use sequential information or LSTM cells as we wish to examine the benefits of framing the regression problem in an image setting.
#We call this method VisualAE, the architecture for which is shown in Figure 5. We used 2D convolutional layers with a kernel size of 5 × 5, stride 2, and padding 2. All layers are followed by ReLU activation and batch normalization. The encoder network consists of 3 convolutional layers which transform a 80 × 80 × 1 input image to 10 × 10 × 512, after which we obtain an embedding vector of length 512 using a fully connected layer. This process is then mirrored for the decoder network, resulting in a forecast image of dimension 80 × 80. We will explain the loss function for training in detail in the next section.
#on veut maintenant créer un autoencoder pour prédire les images

#on va maintenant utiliser visuaAE pour prédire les images
