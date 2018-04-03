#!/usr/bin/env python
# -*- coding=utf-8 -*-

import tensorflow as tf  

filenames = ['feature1', 'feature2', 'uid','label']

filename_queue = tf.train.string_input_producer(filenames, shuffle=False) 
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
record_defaults = [[0], [0], [0], [0]]
col1, col2, col3, col4 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.concat(0, [col1, col2, col3])
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(10):
        example, label = sess.run([features, col4])
        print example.eval(), label.eval()
    coord.request_stop()
    coord.join(threads)




 
