import cv2
import numpy as np
import efficientnet.tfkeras as efn
import tensorflow as tf

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]



def build_model(dim = 256, ef = 0, IMAGE_SIZE=(512,512), NUM_CLASSES=5):
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    
    #x = data_augmentation(inp)
    
    base = EFNS[ef](input_shape=(*IMAGE_SIZE, 3), weights='imagenet', include_top = False)
    
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs = inp,outputs = x)
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    # loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0) 
    
    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def predict(model, filenames):
  pred = model.predict(filenames)
  proba = [tf.nn.softmax(p).numpy().tolist() for p in pred]
  pred = [np.argmax(p) for p in pred]
  return pred, proba

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    
            img = np.stack([img1,img2,img3],axis=-1)
    
        return img

def preprocess_image(image, crop=False, blur = False, sigmaX=10, IMG_PIXEL = 512):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if crop == True:
        image = crop_image_from_gray(image)
    
    image = cv2.resize(image, (IMG_PIXEL, IMG_PIXEL), interpolation = cv2.INTER_AREA)
    
    if blur == True:
        image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    
    return image
