import json
import os
import random

import tensorflow as tf
import numpy as np

BUFFER_SIZE = 1000
dataset_path = 'coco/'


def prepare_repository(train_samples, val_samples, top_k):
    repo = {}
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    train_feature_path_list, train_caption_list = prepare_training_set(
        train_samples)

    # Choose the top k words from the vocabulary
    # after tkoenization cap_list is vector of ints [n_samples, max_length]
    max_length, train_caption_list, tokenizer = tokenize_captions(
        train_caption_list, top_k)

    repo['train_feature_path_list'] = train_feature_path_list
    repo['train_caption_list'] = train_caption_list
    repo['tokenizer'] = tokenizer
    repo['max_length'] = max_length
    repo['start_epoch'] = 0

    val_img_id_list, val_feature_path_list = prepare_validation_set(
        val_samples, 'val')
    repo['val_img_id_list'] = val_img_id_list
    repo['val_feature_path_list'] = val_feature_path_list

    test_img_id_list, test_feature_path_list = data_processing.prepare_validation_set(val_samples, 'test')
    repo['test_img_id_list'] = test_img_id_list
    repo['test_feature_path_list'] = test_feature_path_list

    return repo


def prepare_training_set(n_samples):
    annotations_path = '{}ann2014/annotations/captions_train2014.json'.format(
        dataset_path)
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    image_features_extract_model = prepare_feature_extractor()

    caption_list = []
    feature_path_list = []

    random_n_samples = random.sample(range(1, 82783), n_samples)

    for random_number in random_n_samples:
        annotation = annotations['annotations'][random_number]
        caption = '<start> ' + annotation['caption'] + ' <end>'
        image_id = annotation['image_id']
        image_path = '{}train2014/COCO_train2014_{:012d}.jpg'.format(
            dataset_path, image_id)

        feature_path = extract_features(
            image_path, image_features_extract_model)

        feature_path_list.append(feature_path)
        caption_list.append(caption)

    return feature_path_list, caption_list


def prepare_validation_set(n_samples, data_type):
    annotations_path = '{}ann2014/annotations/captions_{}2014.json'.format(
        dataset_path, data_type)
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    image_features_extract_model = prepare_feature_extractor()

    feature_path_list = []
    image_id_list = []

    random_n_samples = random.sample(range(1, 40504), n_samples)

    for random_number in random_n_samples:
        image_info = annotations['images'][random_number]
        image_path = '{}{}2014/{}'.format(dataset_path, data_type,
                                          image_info['file_name'])
        feature_path = extract_features(
            image_path, image_features_extract_model)

        feature_path_list.append(feature_path)
        image_id_list.append(image_info['id'])

    return image_id_list, feature_path_list


def load_image(image_path):
    # img shape (299,299,3)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    return img


def extract_features(img_name, image_features_extract_model):
    # extract features and stash them in ./coco/train2014_features
    # returns path to the stashed numpy features file
    feature_path = img_name.replace('2014', '2014_features', 1)

    if not os.path.exists(feature_path):
        img = load_image(img_name)
        img_features = image_features_extract_model(tf.expand_dims(img, 0))
        img_features = tf.reshape(img_features, (64, 2048))
        np.save(feature_path, img_features.numpy())

    return '{}.npy'.format(feature_path)


def load_features(features_path):
    img_tensor = np.load(features_path)

    return img_tensor


def prepare_img_tensor(features_path, batch_size):
    # img_tensor size == (batch_size, 64, 2048)
    img_vector = np.zeros((batch_size, 64, 2048), dtype=np.float32)
    for n in range(batch_size):
        img_vector[n, :, :] = load_features(features_path[n])

    return tf.convert_to_tensor(img_vector, dtype=tf.float32)


def prepare_feature_extractor():
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    return image_features_extract_model


def tokenize_captions(train_captions, top_k):
    # Choose the top k words from the vocabulary replace all other words with UNKNOW
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token='<unk>',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)

    # create word-to-index and index-to-word mappings
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    max_length = max(len(t) for t in train_seqs)

    # Pad each vector to the max_length of the captions
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    return max_length, cap_vector, tokenizer
