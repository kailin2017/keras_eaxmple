import numpy
from matplotlib import pyplot
import pandas


class api_image:
    def showImage(image):
        pyplot.gcf().set_size_inches(2, 2)
        pyplot.imshow(image, cmap='binary')
        pyplot.show()

    def showImageLabelPrediction(images, labels, predition, idx=0, num=10):
        pyplot.gcf().set_size_inches(12, 14)
        if num > 25: num = 25
        for i in range(0, num):
            ax = pyplot.subplot(5, 5, 1 + i)
            ax.imshow(images[idx], cmap='binary')
            title = 'label=' + str(labels[idx])
            if len(predition) > 0:
                title += ',predict=' + str(predition[idx])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        pyplot.show()

    def showImageLabelPredictionHot(images, labels, predition, idx=0, num=25):
        pyplot.gcf().set_size_inches(12, 14)
        if num > 25: num = 25
        for i in range(0, num):
            ax = pyplot.subplot(5, 5, 1 + i)
            ax.imshow(numpy.reshape(images[idx], (28, 28)), cmap='binary')
            title = 'label=' + str(numpy.argmax(labels[idx]))
            if len(predition) > 0:
                title += ',predict=' + str(predition[idx])
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        pyplot.show()

    def showImageLabelPredictionDict(images, labels, predition, dict, idx=0, num=10):
        pyplot.gcf().set_size_inches(12, 14)
        if num > 25: num = 25
        for i in range(0, num):
            ax = pyplot.subplot(5, 5, 1 + i)
            ax.imshow(images[idx], cmap='binary')
            title = 'l=' + dict[labels[i][0]]
            if len(predition) > 0:
                title += ',p=' + dict[predition[idx]]
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
        pyplot.show()

    def shoPredicted(image, label, prediction, predicted, dict, i):
        print('\ndata:', i, '-----label:', dict[label[i][0]], ' prediction:', dict[prediction[i]])
        # pyplot.figure(figsize=(2, 2))
        # pyplot.imshow(numpy.reshape(image[i], (32, 32, 3)))
        # pyplot.show()
        for j in range(10):
            print(dict[j], ' Probability:%1.9f' % (predicted[i][j]))

    def showTrainHistory(history, train, validation):
        pyplot.plot(history.history[train])
        pyplot.plot(history.history[validation])
        pyplot.title('Train History')
        pyplot.ylabel(train)
        pyplot.xlabel('Epoch')
        pyplot.legend(['train', 'validation'], loc=' upper left')
        pyplot.show()

    def confusionMatrix(labels, prediction):
        pandas.crosstab(labels, prediction, rownames=['label'], colnames=['predict'])
