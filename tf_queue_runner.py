
import tensorflow as tf

q = tf.FIFOQueue(1000,"float")
counter = tf.Variable(0.0)

increment_op = tf.assign_add(counter, tf.constant(1.0))

enqueue_op = q.enqueue([counter])

qr = tf.train.QueueRunner(q, enqueue_ops = [increment_op, enqueue_op]*1)

#主线程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    enqueue_thread = qr.create_threads(sess, start=True)

    #主线程
    for i in range(10):
        print(sess.run(q.dequeue()))
