import tensorflow as tf
import numpy as np

x = [float(x) for x in range(5 * 15)]
y = tf.reshape(x, [5, 15])

# Use unstack on the 3D matrix with different axis values.
unstack_axis1 = tf.unstack(y, axis=1)

# Use stack to combine tensors.
stack_axis1 = tf.stack(unstack_axis1, axis=1)

# Use unstack on axis value 2.
# unstack_axis2 = tf.unstack(y, axis=2)

stack_axis2 = tf.stack(unstack_axis1, axis=1)

# unstack_axis0 = tf.unstack(y, axis=2)
# stack_axis0 = tf.unstack(y, axis=0)

session = tf.Session()

print("INPUT")
print(session.run(y))
print("UNSTACK AXIS 0")
# ua0 = session.run(unstack_axis0)
# print(ua0)
# print("STACK AXIS 0")
# ua0 = session.run(stack_axis0)
# print(ua0)

print("UNSTACK AXIS 1")
ua1 = session.run(unstack_axis1)
print(ua1)
print("STACK AXIS 1")
print(session.run(stack_axis1))
print("UNSTACK AXIS 2")
# ua2 = session.run(unstack_axis2)
# print(ua2)
# print("STACK AXIS 2")
# print(session.run(stack_axis2))

if np.array_equal(ua0,ua1):
    print("ua0 == ua1")
# if np.array_equal(ua2,ua1):
    # print("ua2 == ua1")
# if np.array_equal(ua2,ua0):
    # print("ua2 == ua0")
