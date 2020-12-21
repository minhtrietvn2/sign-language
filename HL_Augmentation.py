
#
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import cv2
import os
#
count = 0
#A, B, C, D, E, F, G, H, I, <J> , K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, <Z>
name = 'y'
folder_data='E:/project hang/project/data_create/'
folder_temp='E:/project hang/project/temp'
path_2_train = 'E:/project hang/project/data/train/'+name
path_2_test  = 'E:/project hang/project/data/test/'+name
path_2_valid = 'E:/project hang/project/data/valid/'+ name
for cnt in range(23):
    img = load_img(folder_data+name+'/'+name+'_{}.jpg'.format(cnt))
    img = img_to_array(img)
    data = expand_dims(img,0)

    Shift = ImageDataGenerator(width_shift_range=[-20,20])
    Flip = ImageDataGenerator(horizontal_flip=True,vertical_flip=False)
    Rotate = ImageDataGenerator(rotation_range=15)

    gen_s = Shift.flow(data,batch_size=1)
    gen_f = Flip.flow(data, batch_size=1)
    gen_r = Rotate.flow(data,batch_size=1)

    for i in range(10):

        myBacths= gen_s.next()
        myBacthf = gen_f.next()
        myBacthr= gen_r.next()

        img_name = "temp_{}.jpg".format(count);count+=1
        image = myBacths[0].astype('uint8')
        cv2.imwrite(os.path.join(folder_temp, img_name), image)

        img_name = "temp_{}.jpg".format(count);count+=1
        image = myBacthf[0].astype('uint8')
        cv2.imwrite(os.path.join(folder_temp, img_name), image)

        img_name = "temp_{}.jpg".format(count);count+=1
        image = myBacthr[0].astype('uint8')
        cv2.imwrite(os.path.join(folder_temp, img_name), image)

print('Da tao xong Augmentation, tien hang phan chia:')
ik = 0
for cnt2 in range(count):
    cnt5 = 0.8*count
    image_2 = cv2.imread(folder_temp+'/temp_{}.jpg'.format(cnt2))

    cnt3 = str(ik)
    ik+=1
    img_name_2 = name+'_'+cnt3.zfill(3)+'.jpg'
    if      ik < 0.6*count:
        cv2.imwrite(os.path.join(path_2_train, img_name_2), image_2)
    elif    0.6*count <= ik < 0.8*count:
        cv2.imwrite(os.path.join(path_2_test, img_name_2), image_2)
    elif    ik >=0.8*count:
        cv2.imwrite(os.path.join(path_2_valid, img_name_2), image_2)

    os.remove(folder_temp+'/temp_{}.jpg'.format(cnt2))

print('Hoan thanh xong cong viec')