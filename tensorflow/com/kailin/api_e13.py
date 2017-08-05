import re
import os
from pyodbc import Time


class api_e13:
    def filterHtml(text):
        return re.compile(r'<[^>]+>').sub('', text)

    def readSource(path, type):
        filelist = []
        path_pos = path + '/' + type + '/pos/'
        path_neg = path + '/' + type + '/neg/'
        for f in os.listdir(path_pos):
            filelist += [path_pos + f]
        for f in os.listdir(path_neg):
            filelist += [path_neg + f]

        labels = ([1] * 12500 + [0] * 12500)
        texts = []
        for f in filelist:
            with open(f, encoding='utf8') as fileInput:
                texts += [api_e13.filterHtml(" ".join(fileInput.readline()))]
        print('read ', type, ':', len(filelist))

        return texts, labels
