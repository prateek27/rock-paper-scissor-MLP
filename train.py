import numpy as np
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential, save_model
from keras.utils import np_utils
import os
from PIL import Image
from sklearn.utils import shuffle
import pickle

path = './signs/'

signs = os.listdir(path)

x_ , y_ = [], []
train_dict = {}

label = 0
for i in signs:

    if i[0] == '.':
        continue

    images = os.listdir(path + i)
    for j in images:
        if j[0] == '.':
            continue

        img_path = path + i + '/' + j
        img = Image.open(img_path)
        img = np.array(img)
        img = img.reshape( (50,50,1) )
        img = img/255.0
        x_.append(img)
        y_.append( label )

    train_dict[label] = i

    label += 1

print (train_dict)


x = np.array(x_)
y = np.array(y_)
y = np_utils.to_categorical(y)
num_classes = y.shape[1]


x , y = shuffle(x, y, random_state=0)


split = int( 0.6*( x.shape[0] ) )
train_features = x[ :split ]
train_labels = y[ :split ]
test_features = x[ split: ]
test_labels = y[ split: ]

model = Sequential()

model.add( Flatten(input_shape = (50,50,1) ) )

model.add( Dense(128) )
model.add( Activation('relu') )

model.add( Dense(32) )
model.add( Activation('relu') )
model.add( Dropout(0.25) )

model.add( Dense(num_classes) )

model.add( Activation('softmax') )

model.summary()


model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit( train_features, train_labels, validation_data=( test_features, test_labels ), shuffle=True, batch_size=64, nb_epoch=3 )

file = open("train_data.pkl","wb")
pickle.dump(train_dict, file)
file.close()
model.save('model.h5')