#!/usr/bin/env python
# Copyright 2016 Google Inc. All Rights Reserved.
# Copyright 2017 NVIDIA Corp. All Rights Reserved.  
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse

import keras
from keras.applications import ResNet50, VGG16, imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

import math
import multiprocessing

#import mxnet as mx
import numpy as np
from time import time

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def backend_agnostic_compile(model, loss, optimizer, metrics, args):
    if keras.backend._backend == 'mxnet':
        gpu_list = ["gpu(%d)" % i for i in range(args.num_gpus)]
        model.compile(loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            context = gpu_list)
    else:
        if args.num_gpus > 1:
            print("Warning: num_gpus > 1 but not using MxNet backend")
        model.compile(loss=loss,
            optimizer=optimizer,
            metrics=metrics)

def get_synthetic_data(batch_size, image_size=224, nclass=1000):
    n = batch_size
    h = w = image_size
    c = 3
    #images  = (np.random.random(size=(n,c,h,w))*255).astype(np.uint8)
    images  = (np.random.random(size=(n,h,w,c))*255).astype(np.uint8)
    classes = np.zeros((n,nclass), dtype=np.float32)
    classes[:,0] = 1
    last_time = time()
    while True:
        yield images, classes
        current_time = time()
        dt = current_time - last_time
        last_time = current_time
        print (" Images/sec: %.2f" % (batch_size / dt))


def get_python_pipeline_data_2(path='cat.jpg', batch_size=32, reps=100):

    start = time()

    for rep in range(reps):
        batch = np.zeros((batch_size, 480, 480, 3))
        for i in range(batch_size):
            img = load_img('cat.jpg')
            batch[i, :, :, :] = img

    duration = time() - start

    sec_per_im = duration / (batch_size * reps)
    im_per_sec = 1.0 / sec_per_im
    print("Images per second: %.2f" % im_per_sec)
    

def get_python_pipeline_data(batch_size=32, image_size=480, nclass=1000):

    datagen = ImageDataGenerator(
            rotation_range=0,
            width_shift_range=0.0,
            height_shift_range=0.0,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True,
            fill_mode='nearest')

    arr = np.zeros((batch_size, 3, image_size, image_size))
    for i in range(batch_size):
        img = load_img('cat.jpg')
        x = img_to_array(img) 
        x = x.reshape((1,) + x.shape)
        arr[i, :, :, :] = x

    y=np.random.randint(low=0, high=nclass-1, size=batch_size)

    ctr = 0

    start = time() 

    ctr = 0
    for batch in datagen.flow(arr, y, batch_size=1,
        save_to_dir=None, save_prefix=None, save_format=None):
        ctr += 1
        if ctr > 20:
            break 
    end = time()
    sec_im = 1.0 / ((time() - start) / (ctr * batch_size))
    print("%.2f" % sec_im)

# Get data from iterator and report samples per second if desired
def get_data(it, batch_size, report_speed=False, warm_batches_up_for_reporting=100):
    ctr = 0
    warm_up_done = False
    last_time = None

    # Need to feed data as NumPy arrays to Keras
    def get_arrays(db):
        return db.data[0].asnumpy(), to_categorical(db.label[0].asnumpy(),
                                                    nb_classes=args.num_classes)

    # repeat for as long as training is to proceed, reset iterator if need be
    while True:
        try:
            ctr += 1
            db = it.next()

            # Skip this if samples/second reporting is not desired
            if report_speed:

                # Report only after warm-up is done to prevent downward bias
                if warm_up_done:
                    curr_time = time()
                    elapsed =  curr_time - last_time
                    ss = float(batch_size * ctr) / elapsed
                    print(" Batch: %d, Samples per sec: %.2f" % (ctr, ss))

                if ctr > warm_batches_up_for_reporting and not warm_up_done:
                   ctr = 0
                   last_time = time()
                   warm_up_done = True

        except StopIteration as e:
            print("get_data exception due to end of data - resetting iterator")
            it.reset()
            db = it.next()

        finally:
            yield get_arrays(db)

def main():

#    foo = get_python_pipeline_data(batch_size=32, image_size=480, nclass=1000)
#    print(foo)
    get_python_pipeline_data_2()

    parser = argparse.ArgumentParser(
        description="train_resnet50",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #data.add_data_args(parser)
    #data.add_data_aug_args(parser)
    parser.add_argument_group('gpu_config', 'gpu config')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use during training')
    parser.add_argument('--batch-per-gpu', type=int, default=64,
                        help='Batch size per GPU')

    # Use a large augmentation level (needed by ResNet-50)
    #data.set_data_aug_level(parser, 3)

    parser.set_defaults(
        # network
        network          = 'resnet',
        num_layers       = 50,
        # data
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '3,224,224',
        min_random_scale = 1,
        # train
        num_epochs       = 90,
        lr_step_epochs   = '30,60',
        # Assume HyperThreading on x86, only count physical cores
        data_nthreads    = multiprocessing.cpu_count() // 2,
        validation_ex    = 50048
    )

    args = parser.parse_args()

    global_batch_size = args.num_gpus * args.batch_per_gpu
    args.batch_size = global_batch_size

    # Get training and validation iterators
    #train_iter, val_iter = data.get_rec_iter(args)

    #train_gen = get_data(train_iter, batch_size=global_batch_size, report_speed=True)
    #val_gen = get_data(val_iter, batch_size=global_batch_size, report_speed=True)
    train_gen = get_synthetic_data(batch_size=global_batch_size)
    val_gen   = get_synthetic_data(batch_size=global_batch_size)

    it_per_epoch = int(math.ceil(1.0 * args.num_examples / global_batch_size))

    print("Number of iterations per epoch: %d" % it_per_epoch)
    print("Using %d GPUs, batch size per GPU: %d, total batch size: %d" %
          (args.num_gpus, args.batch_per_gpu, global_batch_size))

    gpu_list = ["gpu(%d)" % i for i in range(args.num_gpus)]

    model = ResNet50(weights = None)

    # Print model summary to console
    model.summary()

    # Optimizer (note: we'll be using a learning rate scheduler, see below)
    opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    backend_agnostic_compile(
    model=model, loss='categorical_crossentropy',
    optimizer=opt, metrics=['accuracy'], args=args)

    # Model checkpointer
    checkpointer = ModelCheckpoint(filepath='./resnet_50_weights.hdf5',
                                   verbose=1, save_best_only=True)

    model.fit_generator(generator=train_gen,
                        samples_per_epoch=args.num_examples,
                        nb_epoch=args.num_epochs,
                        verbose=True,
                        callbacks=[checkpointer],
                        validation_data=val_gen,
                        nb_val_samples=args.validation_ex,
                        class_weight=None,
                        max_q_size=2,
                        nb_worker=1,
                        pickle_safe=False,
                        initial_epoch=0)

if __name__ == '__main__':
    main()
