from matplotlib import pyplot
import pandas

class api_image:
    def showImage(image):
        pyplot.gcf().set_size_inches(2,2)
        pyplot.imshow(image,cmap='binary')
        pyplot.show()

    def showImageLabelPrediction(images,labels,predition,idx=0,num=10):
        pyplot.gcf().set_size_inches(12,14)
        if num>25 : num=25

        for i in range(0,num):
            ax = pyplot.subplot(5,5,1+i)
            ax.imshow(images[idx],cmap='binary')
            title = 'label='+str(labels[idx])
            if len(predition)>0:
                title+=',predict='+str(predition[idx])

            ax.set_title(title,fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            idx+=1
        pyplot.show()

    def showTrainHistory(history,train,validation):
        pyplot.plot(history.history[train])
        pyplot.plot(history.history[validation])
        pyplot.title('Train History')
        pyplot.ylabel(train)
        pyplot.xlabel('Epoch')
        pyplot.legend(['train','validation'],loc=' upper left')
        pyplot.show()

    def confusionMatrix(labels,prediction):
        pandas.crosstab(labels,prediction,rownames=['label'],colnames=['predict'])