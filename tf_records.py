
"""
tf.train.Example和tf.train.SequenceExample作为基本单位进行数据读取

tf.train.Example一般用于数值、图像等有固定大小的数据，同时使用Feature指定每个
记录特征的名称和数据类型。
用法如下：
example = tf.train.Example(features = tf.train.Features(feature = {
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))


def _int64_feature (value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature (value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
"""
