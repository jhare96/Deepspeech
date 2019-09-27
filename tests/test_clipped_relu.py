import tensorflow as tf
import os




def relu_clipped(x, clip=20):
    return tf.minimum(tf.maximum(x, 0.0), clip)



def test_relu_clipped():
    config = tf.ConfigProto(allow_soft_placement=True) # Allow automatic CPU placement if GPU kernel not availiable 
    config.gpu_options.allow_growth = True # Allow GPU growth
    sess = tf.Session(config=config)
    x = tf.random.uniform(shape=[5,3])
    y = tf.random.uniform(shape=[3,1])
    print('y', y)
    print('x', x)

    z = relu_clipped(tf.matmul(x,y))
    z2 = tf.minimum(tf.nn.relu(tf.matmul(x,y)), 20)
    sumz = tf.reduce_sum(z)
    yhat = tf.ones(shape=[5,1])

    dydz = tf.gradients(z, y)
    dydz2 = tf.gradients(z2, y)
    dydz, dydz2 = sess.run([dydz, dydz2])
    assert(dydz[0].shape == dydz2[0].shape)
    assert(dydz[0].all() == dydz2[0].all())
    print('dy/dz', dydz, '\ndy/dz2', dydz2)
    

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
test_relu_clipped()