import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D
import os
import matplotlib.pyplot as plt

image_size = (128,128)

def image_prepare(image, size):
    '''
    open, do grayscale, and resize image
    '''
    image1 = cv2.imread(image)
    im_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    im_equal = cv2.equalizeHist(im_gray)
    im_resize = cv2.resize(im_equal, size, interpolation=cv2.INTER_CUBIC)
    
    return im_resize
    
#build list of categories
list_train = os.listdir(f'train/')
y_name = []
len1 = 0
print(list_train)
for n in list_train:
    y_name.append(n)
    list1 = os.listdir(f'train/' + str(n))
    len1 = len1 + len(list1)

#create X_data zeros array
X_data = np.array(np.zeros((len1, image_size[0], image_size[1])))

#create y_data and X_data
w = 0
y_data = []
for c, m in enumerate(list_train):
    list2 = os.listdir(f'train/' + str(m))
    for p in list2:
        im_1 = image_prepare((f'train/' + str(m) + '/' + str(p)), image_size)
        X_data[w] = im_1
        y_data.append(c)
        w+=1
    
print(y_data)
X_norm = X_data/255
y_cat = to_categorical(y_data)
print(y_cat)

# reshape to be:
#[samples][pixels][width][height]
X_norm = X_norm.reshape(X_norm.shape[0], image_size[0], image_size[1], 1).astype('float32')

def convolut_model():
    '''
    build CNN model
    '''
    model = Sequential()
    model.add(Conv2D(64, (5,5), strides=(1,1), activation = 'relu', input_shape=(128, 128, 1)))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))
    
    model.add(Conv2D(128, (5,5), strides=(1,1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(len(list_train), activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    print(model.summary())
    return model

#train CNN model
model = convolut_model()
model.fit(X_norm, y_cat, epochs=20, verbose = 2)
#model.save('cnn_class1.h5')

def test_pic(test):
    '''
    label test images
    '''
    X_test = image_prepare(test, image_size)
    X_t = np.array(np.zeros((1, image_size[0], image_size[1])))
    X_t[0] = X_test
    X_t = X_t.reshape(X_t.shape[0], 128, 128, 1).astype('float32')/255

    pr = model.predict(X_t)
    print(pr)
    cat_ind = np.argmax(pr[0])
    cat_name = y_name[cat_ind]
    dec = ('Na zdjeciu jest ' + str(cat_name) + ' na ' + str(round(pr[0][cat_ind]*100, 0)) + '%.')
    for_visual = cv2.imread(test)
    for_vis = cv2.cvtColor(for_visual, cv2.COLOR_BGR2RGB)
    for_vis = cv2.resize(for_vis, (400, 400), interpolation=cv2.INTER_CUBIC)
    for_vis_text = cv2.putText(img=for_vis,     text=dec, org=(5, 390), color = (0,255,0), fontFace = 4, fontScale = 0.5, thickness = 1)
    plt.figure(figsize=(10,10))
    plt.imshow(for_vis_text)
    plt.show()

#Label test images
list_t = os.listdir('test')
for n in list_t:
    im_t = f'test/' + str(n)
    print(im_t)
    test_pic(im_t)

