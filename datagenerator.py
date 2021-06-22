import os
import random
from PIL import Image
import cv2
import imgaug as ia
import keras
import numpy as np
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from keras.utils import to_categorical
from keras.utils.data_utils import OrderedEnqueuer
from random import choice
from string import ascii_uppercase


class DataGenerator(keras.utils.Sequence):
    def __init__(self, src_dir, img_shape=(416,416), batch_size=4, shuffle=True):#почему размер батча такой маленький + почему он и размер картинки не сходятся со строкой 153
        self.images = []
        self.labels = []
        self.src_dir = src_dir
        #self.src_dir = self.src_dir + '/image'
        #self.scr_dir1 = os.path.join(self.scr_dir, '/out')
        #self.src_dir = os.path.join(self.src_dir, '')
        #self.amount_of_slashes = self.src_dir.count('/')
        self.img_shape = img_shape
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

        #/batch1/
        #       /images/1.jpg
        #       /labels/1.xml

        # /batch1/
        #       /folder/1.jpg
        #       /folder/1.xml

    def __len__(self):
        return int(len(self.images) / float(self.batch_size))

    def on_epoch_end(self):
        src_dir = self.src_dir + '/images'
        src_dir1 = self.src_dir + '/out'
        #print(src_dir1)
        self.images = []
        self.labels = []

        for folder, subs, files in os.walk(src_dir):
            checkfiles = [f for f in os.listdir(folder) if f.endswith(".jpg")]
            #print(checkfiles)
            if len(checkfiles) > 0:
                if self.shuffle:
                    rnd = random.random() * 10000
                    random.Random(rnd).shuffle(checkfiles)
                for f in checkfiles:
                    self.images.append(os.path.join(folder, f))

        for folder, subs, files in os.walk(src_dir1):
            checkfiles1 = [f for f in os.listdir(folder) if f.endswith(".txt")]
            if len(checkfiles1) > 0:
                if self.shuffle:
                    rnd = random.random() * 10000
                    random.Random(rnd).shuffle(checkfiles1)
                for f in checkfiles1:
                    self.labels.append(os.path.join(folder, f))
        if self.shuffle:
            rnd = random.random() * 10000
            random.Random(rnd).shuffle(self.images)
            random.Random(rnd).shuffle(self.labels)

        #self.images = np.array(self.images)
        #self.labels = np.array(self.labels)

#1 0.716797 0.395833 0.216406 0.147222
#0 0.687109 0.379167 0.255469 0.158333
#1 0.420312 0.395833 0.140625 0.166667



    def generate_data(self, indexs):
        src_dir = self.src_dir + '/images'
        src_dir1 = self.src_dir + '/out'
        images_batch = []
        labels_batch = []

        for i in indexs:
            image_name = os.path.join(src_dir, self.images[i])
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)
            labels_strings = open(os.path.join(src_dir1, self.labels[i])).read().splitlines()
            # 1 0.716797 0.395833 0.216406 0.147222
            # cl xcenter ycenter width height
            # img.shape = H, W, C
            bboxes = []

            for line in labels_strings:
                line = line.split(" ")
                xc = float(line[1]) * image.shape[1]
                yc = float(line[2]) * image.shape[0]
                w = float(line[3]) * image.shape[1]
                h = float(line[4]) * image.shape[0]
                label = int(line[0])
                x1 = xc - (w / 2)
                x2 = xc + (w / 2)
                y1 = yc - (h / 2)
                y2 = yc + (h / 2)
                bboxes.append([x1, y1, x2, y2, label])

            ia.seed(1)

            #image = ia.quokka(size=(image.shape[1], image.shape[0]))
            bbs = BoundingBoxesOnImage([
                BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3], label=label) for bbox in bboxes
            ], shape=image.shape)

            seq = iaa.Sequential([
                iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
                iaa.LinearContrast((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05 * 255), per_channel=0.5),
                iaa.Fliplr(0.5)
                #iaa.Crop(percent=(0, 0.src_dir 1)),
            ])

            # Augment BBs and images.
            image_aug, bboxes_aug = seq(image=image, bounding_boxes=bbs)
            labels_strings_ = []
            for j, _ in enumerate(bboxes_aug):
                bbox = bboxes_aug.bounding_boxes[j]
                x1, y1, x2, y2, label = bbox.x1, bbox.y1, bbox.x2, bbox.y2, bbox.label
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                h = y2 - y1
                w = x2 - x1
                string1 = '{0} {1} {2} {3} {4}'.format(str(label), str(xc), str(yc), str(h), str(w))
                #st = str(xc) + str(yc) + str(h) + str(w)
                #print('st: ', st)
                #string1 = labels_strings[i][0] + st

                print('string1: ', string1)
                labels_strings_.append(string1)


            # seq = iaa.Sequential(
            #     [
            #         iaa.Fliplr(0.5),
            #         iaa.Crop(percent=(0, 0.1)),
            #         iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.3))),
            #         iaa.LinearContrast((0.75, 1.5)),
            #         iaa.AdditiveGaussianNoise(loc=0, scale=(0, 0.05 * 255), per_channel=0.5),
            #         iaa.Multiply((0.8, 1.2), per_channel=0.2)
            #     ], random_order=True
            # )
            #seq_det = seq.to_deterministic()
            #augmented_image = seq_det.augment_images([image])
            #augmented_image = augmented_image[0]
            augmented_image = cv2.resize(image_aug, self.img_shape)
            images_batch.append(augmented_image)
            labels_batch.append(labels_strings_)
            #labels_batch.append(to_categorical(self.labels[i], len(self.uniq_classes)))

        images_batch = np.array(images_batch)
        labels_batch = np.array(labels_batch)
        return images_batch, labels_batch

    def __getitem__(self, item):
        indexes = [i + item * self.batch_size for i in range(self.batch_size)]
        a, la = self.generate_data(indexes)
        return a, la

if __name__ == "__main__":

    src_dir = "/home/anastasiia/my_projects/CV_hse"
    #src_dir1 = "/home/anastasiia/my_projects/CV_hse/out"
    train_gen = DataGenerator(src_dir, batch_size=16)
    enqueuer = OrderedEnqueuer(train_gen)
    enqueuer.start(workers=1, max_queue_size=4)
    output_gen = enqueuer.get()
    names = []
    folders = {}
    with open(src_dir + '/class_name.txt', "r") as fi:
        for i, line in enumerate(fi):
            folders[i] = line[:-1]
    gen_len = len(train_gen)
    try:
        for i in range(gen_len):
            batch = next(output_gen)
            for a, la in zip(batch[0], batch[1]):
                k = 0
                while k == 0:
                    name = ''.join([choice(list(ascii_uppercase) + list(map(str, range(0, 10)))) for _ in range(8)])
                    if name not in names:
                        names.append(name)
                        k += 1
                cv2.imwrite(src_dir + '/new_img/' + name + '.jpg', a)
                file = open(src_dir + '/new_out/' + name + '.txt', "w")
                for j in range(len(la)):
                    if j != 0:
                        file.write('\n')
                    file.write(str(la[j]))
                    file.close


                # for j in range(len(la)):
                #     k = 0
                #     while k == 0:
                #         name = ''.join([choice(list(ascii_uppercase) + list(map(str, range(0, 10)))) for _ in range(8)])
                #         if name not in names:
                #             names.append(name)
                #             k += 1
                #     #print(src_dir + '/new_img/' + folders[int(la[j][0])] + '/' + name + '.jpg')
                #     cv2.imwrite(src_dir + '/new_img/' + folders[int(la[j][0])] + '/' + name + '.jpg', a)
                #     file = open(src_dir + '/new_out/' + folders[int(la[j][0])] + '/' + name + '.txt', "w")
                #     file.write(str(la[j]))
                #     file.close

                pass
    finally:
        enqueuer.stop()