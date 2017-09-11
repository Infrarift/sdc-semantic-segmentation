import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

K_KERNEL_REGULARIZER = 1e-3
K_KEEP_PROB = 0.5
K_LEARNING_RATE = 0.0005
K_EPOCHS = 50
K_BATCH_SIZE = 5


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    input_tensor = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    out3_tensor = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    out4_tensor = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    out7_tensor = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob_tensor, out3_tensor, out4_tensor, out7_tensor

tests.test_load_vgg(load_vgg, tf)


def make_conv_1x1(input_tensor, num_classes):
    return tf.layers.conv2d(input_tensor,
                            num_classes,
                            kernel_size=1,
                            strides=(1, 1),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(K_KERNEL_REGULARIZER),
                            padding="same")


def make_conv_transpose(input_tensor, num_classes, scale):
    return tf.layers.conv2d_transpose(input_tensor,
                                      num_classes,
                                      kernel_size=scale * 2,
                                      strides=(scale, scale),
                                      padding="same",
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(K_KERNEL_REGULARIZER))


def make_conv(input_tensor, num_classes):
    return tf.layers.conv2d(input_tensor,
                            num_classes,
                            kernel_size=2,
                            strides=(1, 1),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(K_KERNEL_REGULARIZER),
                            padding="same")


def make_pool(input_tensor):
    return tf.layers.average_pooling2d(input_tensor, 2, 1, padding="same")


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    output = make_conv_1x1(vgg_layer7_out, num_classes)
    output = make_conv_transpose(output, num_classes, 2)

    # Upsample
    pool4 = make_conv_1x1(vgg_layer4_out, num_classes)
    output = tf.add(pool4, output)
    output = make_conv_transpose(output, num_classes, 2)

    # Upsample
    pool3 = make_conv_1x1(vgg_layer3_out, num_classes)
    output = tf.add(pool3, output)
    output = make_conv_transpose(output, num_classes, 2)

    # Scale it up slowly to be sharper.
    output = make_conv(output, num_classes)
    output = make_pool(output)
    output = make_conv_transpose(output, num_classes, 2)

    output = make_conv(output, num_classes)
    output = make_pool(output)
    output = make_conv_transpose(output, num_classes, 2)

    return output

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for image, label in get_batches_fn(batch_size):
            feed = {
                input_image: image,
                correct_label: label,
                keep_prob: K_KEEP_PROB,
                learning_rate: K_LEARNING_RATE
            }
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed)
        print("Epoch {}. Loss: {}".format(e, loss))

tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    correct_label_tensor = tf.placeholder("uint8", None)
    learn_rate_tensor = tf.placeholder("float", None)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_tensor, keep_prob_tensor, out3_tensor, out4_tensor, out7_tensor = load_vgg(sess, vgg_path)
        final_output = layers(out3_tensor, out4_tensor, out7_tensor, num_classes)
        logits, train_op, cross_entropy_loss = optimize(final_output,
                                                        correct_label_tensor,
                                                        learn_rate_tensor,
                                                        num_classes)

        train_nn(sess,
                 epochs=K_EPOCHS,
                 batch_size=K_BATCH_SIZE,
                 get_batches_fn=get_batches_fn,
                 train_op=train_op,
                 cross_entropy_loss=cross_entropy_loss,
                 input_image=input_tensor,
                 correct_label=correct_label_tensor,
                 keep_prob=keep_prob_tensor,
                 learning_rate=learn_rate_tensor
                 )

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_tensor)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
