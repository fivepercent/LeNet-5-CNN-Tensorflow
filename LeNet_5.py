import util
import tensorflow as tf

def classifier(x):
	#parameters for convolution kernels
	IMG_DEPTH=1
	C1_KERNEL_SIZE, C2_KERNEL_SIZE, C3_KERNEL_SIZE=5,5,5
	C1_OUT_CHANNELS,C2_OUT_CHANNELS,C3_OUT_CHANNELS=6,16,120
	C1_STRIDES,C2_STRIDES,C3_STRIDES=1,1,1

	P1_SIZE,P2_SIZE=2,2
	P1_STRIDE,P2_STRIDE=2,2

	F4_SIZE,F5_SIZE=84,10

	C1_kernel=util.weights([C1_KERNEL_SIZE,C1_KERNEL_SIZE,IMG_DEPTH, C1_OUT_CHANNELS],0.1,'C1_kernel')
	C2_kernel=util.weights([C2_KERNEL_SIZE,C2_KERNEL_SIZE, C1_OUT_CHANNELS, C2_OUT_CHANNELS],0.1,'C2_kernel')
	C3_kernel=util.weights([C3_KERNEL_SIZE,C3_KERNEL_SIZE, C2_OUT_CHANNELS, C3_OUT_CHANNELS],0.1,'C3_kernel')

	C1_bias=util.bias([C1_OUT_CHANNELS], 'C1_bias')
	C2_bias=util.bias([C2_OUT_CHANNELS], 'C2_bias')
	C3_bias=util.bias([C3_OUT_CHANNELS], 'C3_bias')

	#LeNet-5 structure
	C1=util.convLayer(x, C1_kernel, C1_STRIDES, 'SAME')
	ReLU1=tf.nn.relu(C1+C1_bias)
	P1=util.max_pool(ReLU1,P1_SIZE, P1_STRIDE)

	C2=util.convLayer(P1, C2_kernel, C2_STRIDES, 'SAME')
	ReLU2=tf.nn.relu(C2+C2_bias)
	P2=util.max_pool(ReLU2,P2_SIZE, P2_STRIDE)

	C3=util.convLayer(P2, C3_kernel, C3_STRIDES, 'SAME')
	ReLU3=tf.nn.relu(C3+C3_bias)
	
	num_F4_in=(int)(ReLU3.shape[1]*ReLU3.shape[2]*ReLU3.shape[3])
	F4_in=tf.reshape(ReLU3,[-1,num_F4_in])

	F4_weights=util.weights([num_F4_in, F4_SIZE],0.1,'F4_weights')
	F4_bias=util.bias([F4_SIZE],'F4_bias')
	F4=tf.matmul(F4_in, F4_weights)
	ReLU4=tf.nn.relu(F4+F4_bias)

	F5_weights=util.weights([F4_SIZE, F5_SIZE],0.1,'F5_weights')
	F5_bias=util.bias([F5_SIZE],'F5_bias')
	F5=tf.matmul(ReLU4, F5_weights)+F5_bias

	return F5
