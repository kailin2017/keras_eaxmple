from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, ZeroPadding2D
from com.kailin.api_file import api_file
from com.kailin.api_image import api_image

dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
        9: 'truck'}
(train_image, train_label), (test_image, test_label) = cifar10.load_data()

print('train image     length:', len(train_image), ' shape:', train_image.shape)
print(' test  image     length:', len(test_image), ' shape:', test_image.shape)

train_image_hot = train_image.astype('float32') / 255
test_image_hot = test_image.astype('float32') / 255

train_label_hot = np_utils.to_categorical(train_label)
test_label_hot = np_utils.to_categorical(test_label)

model = Sequential()
model.add(Conv2D(input_shape=(32, 32, 3), filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(units=2048, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=768, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10, activation='softmax'))

try:
    api_file.loadMode(model, 'E:\pythonwork\data\e09.h5')
except:
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_image_hot, train_label_hot, validation_split=0.2, batch_size=128, epochs=10, verbose=2)
    api_image.showTrainHistory(history, 'acc', 'val_acc')
    api_image.showTrainHistory(history, 'loss', 'val_loss')
    scores = model.evaluate(test_image_hot, test_label_hot, verbose=0)
    print('\n', scores, '\n')
    api_file.saveMode(model, 'E:\pythonwork\data\e09.h5')

result1 = model.predict_classes(test_image_hot)
result2 = model.predict(test_image_hot)

api_image.showImageLabelPredictionDict(test_image, test_label, result1, dict, 0, 25)
for i in range(100, 150):
    api_image.shoPredicted(test_image, test_label, result1, result2, dict, i)
