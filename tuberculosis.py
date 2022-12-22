import tensorflow as tf

classes = ['Normal', 'Tuberculosis']
IMAGE_SHAPE = (96, 96)
model = tf.keras.models.load_model("TB")


def load_and_prep_tb(image):
    """
    Reads an image from filename, turns it into a tensor and reshapes it to (img_shape, img_shape,, color_channels)
    """
    # Read in the image
    image = tf.io.read_file(image)
    # Decode the read file into a tensor
    image = tf.image.decode_image(image)
    # Resize the image
    image = tf.image.resize(image, size=IMAGE_SHAPE)
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    return image


def pred_model_tb(imgpath):
    img_2 = load_and_prep_tb(imgpath)

    with tf.device('/cpu:0'):
        pred_prob = model.predict(tf.expand_dims(img_2, axis=0))
        pred_class = classes[pred_prob.argmax()]

    return pred_class, pred_prob.max()


# img_path = "testing_input/tuberculosis/CHNCXR_0007_0.png"
# class_result, prob_result = pred_model_tb(img_path)
# predictions = (class_result, int(prob_result * 100))
#
# print(predictions)
