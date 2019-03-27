import tensorflow as tf

m1 = tf.constant([[3, 3]])
m2 = tf.constant([[1], [2]])

product = tf.matmul(m1, m2)

x = tf.Variable([2, 3])
y = tf.constant([1, 2])

sub = tf.subtract(x,y)
sun = tf.add(x,sub)
state = tf.Variable(0, name='ewqeer')
new_value = tf.add(state, 1)
update = tf.assign(state,new_value)


inpout1 = tf.placeholder(tf.float32)
inpout2 = tf.placeholder(tf.float32)
output = tf.multiply(inpout1,inpout2)



init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        sess.run(update)
        print(sess.run(state))

    print(sess.run(output,feed_dict={inpout1: [[3, 5],[8,7]], inpout2: [[2],[4]]}))




