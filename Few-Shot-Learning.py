"""
Implementation of Contrastive Loss.
Users are free to copy and distribute only with citation.
https://github.com/ShravanAnandk7/Keras-Image-Embeddings-using-Contrastive-Loss
Last updated 09 Jan 2022
TODO: 1) Add cosine distance metric
      2) Add Batch-Hard and Semi-Hard triplet generation
      3) Resize with padding in pre-processing pipe
"""
#%%
import os
import numpy as np
import pandas as pd
from functools import partial
from cv2 import cv2
import tensorflow as tf
import random
import itertools
import tensorflow.keras.utils as KU
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.losses as KLo
import tensorflow.keras.optimizers as KO
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from imgaug import augmenters as arg
BASE_DIR    = os.path.dirname(__file__)
os.chdir(BASE_DIR)
# PARAMTERS
MODEL_DIR       =  os.path.join(BASE_DIR,"models")
DATASET_DIR     =  os.path.join(BASE_DIR,"datasets")
BATCH_SIZE      =  10
NUM_EPOCHS      =  1
INPUT_SHAPE     =  299
EMBEDDING_SIZE  =  32
LOSS_MARGIN     =  0.4
HUBER_DELTA     =  0.5
# Edit the type of augmentation required specific 
# for your data here
AUGMENTATION      = arg.Sequential(

                            [       
                                arg.OneOf([arg.Fliplr(0.5), arg.Flipud(0.5)]),
                                arg.Affine(scale = (0.85, 1.05),name="scale"),
                                arg.Rotate(rotate = (-180,180),name = "1a2_rotate_1"),
                                arg.TranslateX(percent = (-0.05, 0.05), name= "1a3_translatex_1"),
                                arg.TranslateY(percent = (-0.05, 0.05), name= "1a4_translatey_1"),
                                arg.OneOf([
                                        arg.Sometimes(0.9,arg.MultiplyAndAddToBrightness(mul=(0.70, 1.30), add=(-5, 5)),name="2a1_MulAddBrightness"),
                                        arg.MultiplySaturation(mul=(0.95,1.05),name="2b3_MulSat"),
                                        arg.MultiplyAndAddToBrightness(mul=(1,1.5), add=(-10,10),name="2b4_MulAddBrightness")
                                            ]),
                                arg.Sometimes(0.2,arg.GaussianBlur(sigma = (0.0, 1.5)),name="3a1_gaussian_blur_0.2")
                            ]
                            )
class FewShotTripletDataGen(KU.Sequence):
    def __init__(self,path,image_dim, batch_size = 1, shuffle = True,
                 augmenter = None):
        self.image_dim  = image_dim
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.augmenter  = augmenter
         
        categories = os.listdir(path)
        folder_paths = list(map(partial(os.path.join,path),categories))
        images = list(map(os.listdir, folder_paths))
        self.dataframe = pd.DataFrame(
            {
                "categories" :categories,
                "folder path" : folder_paths,
                "images": images
            })
        print("Categories found",self.dataframe.__len__())
        self.triplets   = list(itertools.permutations(np.arange(len(
                          self.dataframe)),2))
        self.on_epoch_end()
        print("Total triplets : ",len(self.triplets))

    def __len__(self):
        return int(np.floor(len(self.triplets) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.triplets))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        """
        Outputs = [Anchor, Positive, Negative] with 
        shape (Batch, 3, Height, Width, 3)
        """
        batch_indexes = self.indexes[index*self.batch_size:(index+1)
                        *self.batch_size]
        X, y = self.__batch_all_triplet_data_gen(batch_indexes) 
        return X, y
    
    def __batch_all_triplet_data_gen(self,batch_indexes):
        X=[]
        anchor_list = []
        positive_list = []
        negative_list = []
        # print("batch Indices : ", batch_indexes)
        for row_id in batch_indexes:
            anchor, positive = [os.path.join(self.dataframe.loc[self.triplets[row_id][0]]["folder path"],i) for i in random.sample(self.dataframe.loc[self.triplets[row_id][0]]["images"],2)]
            # # anchor = os.path.join(self.dataframe.loc[self.triplets[row_id][0]]["folder path"],random.choice(self.dataframe.loc[self.triplets[row_id][0]]["images"]))
            # positive = os.path.join(self.dataframe.loc[self.triplets[row_id][0]]["folder path"],random.choice(self.dataframe.loc[self.triplets[row_id][0]]["images"]))
            negative = os.path.join(self.dataframe.loc[self.triplets[row_id][1]]["folder path"],random.choice(self.dataframe.loc[self.triplets[row_id][1]]["images"]))
            # print(anchor,'\n',positive,'\n',negative)

            anchor = self.pre_process(self.__augmenter(cv2.imread(anchor)))
            positive = self.pre_process(self.__augmenter(cv2.imread(positive)))
            negative = self.pre_process(self.__augmenter(cv2.imread(negative)))
            # # print(anchor.shape, positive.shape, negative.shape)
            anchor_list.append(anchor)
            positive_list.append(positive)
            negative_list.append(negative)
        return (np.asarray(anchor_list),np.asarray(positive_list),np.array(negative_list)), None
    
    def pre_process(self,image):
        """ 
        Model specific image preprocessing function
        TODO: Resize with crop and padding
        """
        image = cv2.resize(image,self.image_dim)
        image = image/127.5 -1
        return image
    
    def __augmenter(self,image):
        if self.augmenter is not None:      
            image_shape = image.shape
            image = self.augmenter.augment_image(image)
            #Augmentation shouldn't change image size
            assert image.shape == image_shape
        return image
class TripletLossLayer(KL.Layer):
    def __init__(self,margin=1,delta=1,**kwargs):
        self.margin = margin
        self.huber_delta = delta
        super(TripletLossLayer, self).__init__(**kwargs)
        pass
    def euclidean_distance(self,x,y):
        """
        Euclidean distance metric
        """
        return K.sum(K.square(x-y), axis=-1)
    def cosine_distance(self,x,y):
        """
        Cosine distance metric
        """
        pass
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = self.euclidean_distance(anchor[0],positive[0])
        n_dist = self.euclidean_distance(anchor[0],negative[0])
        t_loss  = K.maximum(p_dist - n_dist + self.margin, 0)
        # Huber loss
        L1_loss = K.switch(t_loss < self.huber_delta, 0.5 * t_loss ** 2, self.huber_delta * (t_loss - 0.5 * self.huber_delta))
        return K.sum(L1_loss)
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
#%% Build network model
def base_network():
    """
    Base CNN model trained for embedding extraction
    """
    return( 
            KM.Sequential(
                [   
                    KL.Input(shape=(INPUT_SHAPE,INPUT_SHAPE,3)),
                    KL.Conv2D(8,(3,3)),
                    KL.ReLU(),
                    KL.MaxPool2D(pool_size=(1,2)),
                    # KL.BatchNormalization(),
                    KL.Conv2D(16,(3,3)),
                    KL.ReLU(),
                    KL.MaxPool2D(pool_size=(2,1)),
                    KL.BatchNormalization(),
                    KL.Conv2D(32,(3,3)),
                    KL.ReLU(),
                    KL.MaxPool2D(pool_size=(1,1)),
                    KL.GlobalAveragePooling2D(),
                    # Don't Change the below layers
                    KL.Dense(EMBEDDING_SIZE,activation = 'relu'),
                    # KL.Lambda(lambda x: K.l2_normalize(x,axis=-1))
                ]))
base = base_network()
# Optional to load weights from trained model
base.load_weights(os.path.join(BASE_DIR, "models","few-shot.h5"))
print(base.summary())
def triplet_network(base):
    Anchor   = KL.Input(shape=(INPUT_SHAPE,INPUT_SHAPE,3),name= "anchor_input")
    Positive = KL.Input(shape=(INPUT_SHAPE,INPUT_SHAPE,3),name= "positive_input")
    Negative = KL.Input(shape=(INPUT_SHAPE,INPUT_SHAPE,3),name= "negative_input")

    Anchor_Emb = base(Anchor)
    Positive_Emb = base(Positive)
    Negative_Emb = base(Negative)
    
    loss = TripletLossLayer(LOSS_MARGIN,HUBER_DELTA)([Anchor_Emb,Positive_Emb,Negative_Emb])
    model = KM.Model(inputs = [Anchor,Positive,Negative], outputs=loss)
    return model
#%% Train Model
triplet_model = triplet_network(base)
optimizer = KO.Adam(lr = 0.001)
triplet_model.compile(loss=None,optimizer=optimizer)
print("Train Data :")
train_gen = FewShotTripletDataGen(path = os.path.join(
             DATASET_DIR,"few-shot-dataset","train"),
             image_dim=(INPUT_SHAPE,INPUT_SHAPE), 
             batch_size=BATCH_SIZE,augmenter=AUGMENTATION)
print("Test Data :")

valid_gen = FewShotTripletDataGen(path = os.path.join(
             DATASET_DIR,"few-shot-dataset","test"),
             image_dim=(INPUT_SHAPE,INPUT_SHAPE), 
             batch_size=BATCH_SIZE)
triplet_model.fit(x=train_gen,
                  batch_size=BATCH_SIZE,
                  validation_data=valid_gen,
                  epochs=NUM_EPOCHS,
                  workers=1)
# Save trained model weights
base.save_weights(os.path.join(BASE_DIR, "models","few-shot.h5"))
#%% Prediction with trained base model
image_path = os.path.join(
             DATASET_DIR,"few-shot-dataset","test","cat","0013.jpg")
print(image_path)
input = train_gen.pre_process(cv2.imread(image_path))
output_embeddings = base.predict(np.expand_dims(input,axis=0))
print(output_embeddings)
