import os
import cv2
import numpy as np
import keras
from keras.utils import to_categorical
from keras.utils.data_utils import OrderedEnqueuer


class DataGenerator(keras.utils.Sequence):

    def __init__(self, src_dir, img_shape = (256, 256), batch_size = 4, shuffle = True, class_name = 'class_name.txt'):
        self.images = []
        self.labels = []
        self.src_dir = src_dir
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.shuffle = shuffle
        self.class_name = open(class_name).read().splitlines()


    def __len__(self):
        return int(len(self.images) / float(self.batch_size))


    def generate_data(self, indexs):

        images_batch = []
        labels_batch = []

        for i in indexs:
            image_name = os.path.join(self.scr_dir, self.images[i])
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)
            image = cv2.resize(image, self.img_shape)
            image /= 255.

            images_batch.append(image)
            labels_batch.append(to_categorical(self.labels[i], len(self.class_num)))

        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)

        return images_batch, labels_batch

    def __getitem__(self, item):
        indexes = [i + item * self.batch_size for i in range(self.batch_size)]
        a, la = self.generate_data(indexes)
        return a, la

if __name__ == '__main__':

    src_dir = '/home/anastasiia/my_projects/CV_hse'
    train_gen = DataGenerator(src_dir, batch_size= 16, class_name= '/home/anastasiia/my_projects/CV_hse/class_name.list')
    enquever = OrderedEnqueuer(train_gen)
    enquever.start(workers= 1, max_queue_size= 4)
    output_gen = enquever.get()

    gen_len = len(train_gen)
    try:
        for i in range(gen_len):
            batch = next(output_gen)
            for a, la in zip(batch[0], batch[1]):
                print(a.shape)
                cv2.imshow('win', a)
                print(la)
                print(np.argmax(la))
                cv2.waitKey(0)
    finally:
        enquever.stop()


