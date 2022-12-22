import tensorflow as tf


def load_and_prep_bt(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, (224, 224))

    return img


class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']


def pred_model_bt(filepath):
    img = load_and_prep_bt(filepath)
    model = tf.keras.models.load_model("Brain_tumor")

    with tf.device('/cpu:0'):
        pred_prob = model.predict(tf.expand_dims(img, axis=0))
        pred_class = class_names[pred_prob.argmax()]

    return pred_class, pred_prob.max()

# filepath = "testing_input/brain_tumor/no_tumor.jpg"
# print(pred_model_bt(filepath))


