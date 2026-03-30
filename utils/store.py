import os
import glob
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation

PATH = '/mnt/Data/jwandas/Data/tiger-training/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/'
filenames = glob.glob('*.xml', root_dir=PATH)


an_set = set()
k = 1
for file in filenames:
    file = PATH + file
    an = WholeSlideAnnotation(file)
    [an_set.add(el) for el in an.labels.names]
    print(k)
    k+=1

print()