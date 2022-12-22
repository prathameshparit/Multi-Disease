import tensorflow as tf

classes = ['Cataract', 'Normal']
model = tf.keras.models.load_model("Cataract")
# Function to load and preprocess an image
IMG_SIZE = (224, 224)


def load_and_prep_ct(filepath):
    img = tf.io.read_file(filepath)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, IMG_SIZE)

    return img


def pred_model_ct(imgpath):
    img_2 = load_and_prep_ct(imgpath)

    pred_prob = model.predict(tf.expand_dims(img_2, axis=0))
    print(pred_prob)
    pred_class = classes[pred_prob.argmax()]

    return pred_class, pred_prob.max()


# img_path = "testing_input/cataract_input/cataract_021.png"
# class_result, prob_result = pred_model_ct(img_path)
# predictions = (class_result, int(prob_result * 100))
#
# print(predictions)
