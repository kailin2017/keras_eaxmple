from keras.datasets import cifar10



(train_image,train_label),(test_image,test_label) = cifar10.load_data()

print('train image     length:',len(train_image),' shape:',train_image.shape)
print(' test  image     length:',len(test_image),' shape:',test_image.shape)