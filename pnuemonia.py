import tensorflow as tf

classes = ['Normal', 'Pneumonia']
IMAGE_SHAPE = (224, 224)
model = tf.keras.models.load_model("Pneumonia")

def load_and_prep_image(image):
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

def pred_model(imgpath):
    img_2 = load_and_prep_image(imgpath)

    with tf.device('/cpu:0'):
        pred_prob = model.predict(tf.expand_dims(img_2, axis=0))
        pred_class = classes[pred_prob.argmax()]

    return pred_class, pred_prob.max()

# img_path = "testing_input/pneumonia_images/img3.jpeg"
# class_result, prob_result = pred_model(img_path)
# predictions = (class_result, int(prob_result * 100))
#
# print(predictions)