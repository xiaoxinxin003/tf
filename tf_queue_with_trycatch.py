
import tensorflow as tf

q = tf.FIFOQueue(1000,"float")
counter = tf.Variable(0.0)

increment_op = tf.assign_add(counter, tf.constant(1.0))

enqueue_op = q.enqueue([counter])

qr = tf.train.QueueRunner(q, enqueue_ops = [increment_op, enqueue_op]*1)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
#coordinator协调器，可以看作一种信号量，用来做同步
coord = tf.train.Coordinator()

enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

coord.request_stop()#通知其他线程关闭

#主线程
for i in range(0, 10):
    try:
        print(sess.run(q.dequeue()))
    except tf.errors.OutOfRangeError:
        break

coord.join(enqueue_threads) #直到其他线程结束才退出
