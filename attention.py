import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
import numpy as np
from nets import resnet_v1
from preprocessing import vgg_preprocessing
import os
import matplotlib.pyplot as plt

def resnet_101(inputs,num_classes=2,scope='resnet_v1_101',is_training=True,output_stride=16):
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net = resnet_v1.resnet_v1_101(inputs,num_classes=2,is_training=is_training,global_pool=tf.AUTO_REUSE,reuse=tf.AUTO_REUSE,scope=scope)
    return net


tf.reset_default_graph()

file_name = 'food.jpg'
test_time = 500
total_time = 500

checkpoint_file = 'model/model.ckpt-142730'

current = 0
original_image = tf.image.decode_jpeg(tf.read_file(file_name), channels=3)
temp_im = Image.open(file_name)
temp_arr = np.array(temp_im)
sal = np.zeros(temp_arr.shape[:-1])
total_sal = np.zeros(temp_arr.shape[:-1])

while current < total_time:
    print(current)
    current += test_time
    image_size = resnet_v1.resnet_v1_101.default_image_size #  299
    processed_images = tf.zeros([test_time,image_size,image_size,3],dtype=tf.uint8)
    im = []
    masks = []
    h = 20
    w = 20 
    p = 6
    for t in range(test_time):
        #mask_grid = tf.constant([np.random.randint(low=8,high=21),np.random.randint(low=8,high=21)])
        mask1 = tf.random_uniform((h,w),0,p,dtype=tf.int32)
        comparison = tf.equal(mask1,tf.constant(0))
        mask1 = tf.where(comparison,tf.constant(0,shape=(h,w)),tf.constant(1,shape=(h,w)))    
        mask = tf.stack([mask1,mask1,mask1],axis=2)
        Ch = tf.shape(original_image)[0] // h
        Cw = tf.shape(original_image)[1] // w
        mask = tf.image.resize_images(mask,(Ch*(h+1),Cw*(w+1)),method=tf.image.ResizeMethod.BILINEAR)
        mask = tf.random_crop(mask,tf.shape(original_image))
        mask = tf.cast(mask,tf.uint8)
        image = tf.multiply(original_image,mask)
        processed_image = vgg_preprocessing.preprocess_image(image,image_size,image_size,is_training=False,)
        im.append(processed_image)
        masks.append(mask[:,:,0])
    masks = tf.stack(masks)
    processed_images = tf.stack(im)
    logits, end_points = resnet_101(processed_images, is_training=False)   

    probabilities = tf.nn.softmax(logits)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_file)

        [masks2,probabilities2] = sess.run([masks,probabilities])
        mask = masks2
        for i in range(mask.shape[0]):
            sal += probabilities2[i][0] * mask[i,:,:]
            total_sal += mask[i,:,:]

sal = np.divide(sal,total_sal)
img = Image.open(file_name)
plt.imshow(img)
plt.imshow(np.array(sal), alpha=0.5, cmap='jet')
plt.colorbar()
fig = plt.gcf()
plt.margins(0,0)
fig.savefig('map.jpg', dpi=500, bbox_inches='tight')

