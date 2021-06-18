import os
import re
import csv
import h5py
import glob
import shutil
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as B


num_classes = [2, 1] # grapy, edges

def extract_edges(mask):
    edges = tf.image.sobel_edges(mask)

    boundary = tf.zeros([256, 192, 1])
    for i in range(edges.shape[0]):
        region_bdry = edges[i]
        img = tf.reduce_sum(region_bdry ** 2, axis=-1) + 1e-12
        img = tf.math.sqrt(img)
        boundary += tf.cast(img>0.05, tf.float32)

    boundary = tf.cast(boundary>0, tf.float32)

    return boundary

def filterMask(mask, valList):
    newmask = tf.cast(mask == valList[0], tf.float32)
    for val in valList[1:]:
        newmask += tf.cast(mask == val, tf.float32)
    return newmask

def simulate_misalign(mask):        
    mask = tf.expand_dims(mask, axis=0)

    k = tf.random.uniform([])
    if k < 0.16:
        kernel = tf.cast(tf.ones((2, 3, 1)), tf.double)
    elif k < 0.32:
        kernel = tf.cast(tf.ones((2, 2, 1)), tf.double)
    elif k < 0.48:
        kernel = tf.cast(tf.ones((3, 2, 1)), tf.double)
    elif k < 0.64:
        kernel = tf.cast(tf.ones((1, 2, 1)), tf.double)
    elif k < 0.80:
        kernel = tf.cast(tf.ones((2, 1, 1)), tf.double)
    else:
        kernel = tf.cast(tf.ones((1, 1, 1)), tf.double)

    eroded_mask = tf.nn.erosion2d(
            value=tf.cast(mask, tf.double),
            filters=kernel,
            strides=(1, 1, 1, 1),
            dilations=(1, 3, 3, 1),
            padding="SAME",
            data_format="NHWC",
    )
    eroded_mask = eroded_mask[0] + 1
    eroded_mask = tf.cast(eroded_mask, tf.float32)

    # To account for misalignment of grapy
    eroded_part = mask[0] - eroded_mask
    eroded_part_cloth = eroded_part

    return eroded_mask, eroded_part_cloth


def erode_grapy(mask):        
    mask = tf.expand_dims(mask, axis=0)

    kernel = tf.cast(tf.ones((2, 2, 1)), tf.double)

    eroded_mask = tf.nn.erosion2d(
            value=tf.cast(mask, tf.double),
            filters=kernel,
            strides=(1, 1, 1, 1),
            dilations=(1, 3, 3, 1),
            padding="SAME",
            data_format="NHWC",
    )
    eroded_mask = eroded_mask[0] + 1
    eroded_mask = tf.cast(eroded_mask, tf.float32)

    return eroded_mask

def dilate_skin(mask):
    mask = tf.expand_dims(mask, axis=0)

    kernel = tf.cast(tf.ones((2, 2, 1)), tf.double)

    dilated_mask = tf.nn.dilation2d(
            input=tf.cast(mask, tf.double),
            filters=kernel,
            strides=(1, 1, 1, 1),
            dilations=(1, 3, 3, 1),
            padding="SAME",
            data_format="NHWC"
    )
    dilated_mask = dilated_mask[0] - 1
    dilated_mask = tf.cast(dilated_mask, tf.float32)

    return dilated_mask

def prep_cloth_random(clothMask, cloth, person_list, k=-1):

    if k == -1:
        k = tf.random.uniform([])

    #if k < 0.05:
    #    i_cloth_ten = tf.ones([256, 192, 3])
    if k < 0:
        i_cloth_ten = tf.random.uniform([256, 192, 3], minval=-1, maxval=1)
    elif k < 0.5:
        [person_with_bg, xmin, ymin, xmax, ymax] = person_list
        #tf.print(xmin, ymin, xmax, ymax)
        
        if tf.random.uniform([]) < 0.5:
            x = 1
            y = tf.maximum(ymin-15, 0)
            height = tf.minimum(xmin+10, 254-x)
            width = tf.minimum(ymax-ymin+10, 190-y)
        else: 
            x = 1
            y = tf.maximum(ymin-15, 0)
            height = tf.maximum(xmax-x, 10)
            width = tf.minimum(30, 190-y)

        i_cloth_ten = tf.image.crop_to_bounding_box(person_with_bg, x, y, height, width)

        i_cloth_ten = tf.image.resize(i_cloth_ten, [256, 192], method='nearest')
    else:
        px = tf.where(clothMask == 1)

        try:
            # Crop Cloth for accurate scaling
            x = tf.cast(tf.minimum(tf.maximum(tf.reduce_min(px[:, 0]), 10), 254), tf.int32)
            y = tf.cast(tf.minimum(tf.maximum(tf.reduce_min(px[:, 1]), 10), 190), tf.int32)
            height = tf.cast(tf.minimum(tf.maximum(tf.reduce_max(px[:, 0]) - tf.reduce_min(px[:, 0]), 10), 244), tf.int32)
            width = tf.cast(tf.minimum(tf.maximum(tf.reduce_max(px[:, 1]) - tf.reduce_min(px[:, 1]), 10), 180), tf.int32)
            #tf.print(x, y, height, width, 'before')

            x = tf.minimum(x, 254-height)
            y = tf.minimum(y, 190-width)
            #tf.print(x, y, height, width, 'after')

            i_cloth_ten = tf.image.crop_to_bounding_box(cloth, x, y, height, width)

            i_cloth_ten = tf.image.resize(i_cloth_ten, [256, 192], method='nearest')
        except:
            i_cloth_ten = tf.ones([256, 192, 3])

    return i_cloth_ten

def curvy_cut(cols):
    """
    Code to produce random curvy cut below the sleeve
    """
    image = tf.random.uniform(shape=(1, 600, 1))
    image = B.concatenate((tf.zeros((1, 600, 1)), image), 0)
    image = B.concatenate((image, tf.ones((1, 600, 1))), 0)
    image = tf.image.resize(image, [30, 10000])
    image = tf.cast(image + 0.5, tf.int32)

    diff = tf.cast(tf.random.uniform(shape=()) * 25.0 + 50.0, tf.int64)
    px = tf.where(cols == 1)
    y1 = tf.reduce_min(px[:, 1])
    y2 = tf.reduce_max(px[:, 1])
    y1 = tf.maximum(tf.minimum(y1 - 1, 190), 1)
    y2 = tf.maximum(tf.minimum(y2 - 1, 191), y1 + 1)
    x = tf.reduce_min(tf.where(cols)[:, 0])
    y = tf.maximum(tf.minimum(x + diff, 254), 1)
    y_coord = tf.cast(tf.random.uniform(shape=()) * tf.cast((10000 - (y2 - y1)), tf.float32), tf.int64)
    k = tf.maximum(tf.minimum(tf.cast(10, tf.int64), y), 1)
    sliced = tf.slice(image, [20-k, y_coord, 0], [k, y2 - y1, 1])

    hand_mask = tf.cast(
        B.concatenate(
            (
                B.concatenate(
                    (
                        tf.ones([y, y1, 1]),
                        B.concatenate((tf.zeros([y - k, y2 - y1, 1]), tf.cast(sliced, tf.float32)), axis=0),
                        tf.ones([y, 192 - y2, 1]),
                    ),
                    1,
                ),
                tf.ones([256 - y, 192, 1]),
            ),
            0,
        ),
        tf.float32,
    )

    return hand_mask

def _filter(data, model_inputs):
    string = data["category"]
    # return True
    return tf.strings.regex_full_match(string, "real-background")

def _parse(proto):
    keys_to_features = {
        "personno": tf.io.FixedLenFeature([], tf.int64),
        "person": tf.io.FixedLenFeature([], tf.string),
        "personMask": tf.io.FixedLenFeature([], tf.string),
        "densepose": tf.io.FixedLenFeature([], tf.string),
        "is_shorts": tf.io.FixedLenFeature([], tf.string),
        "is_short_sleeves": tf.io.FixedLenFeature([], tf.string),
        "category": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    person = (tf.cast(tf.image.decode_jpeg(parsed_features["person"], channels=3), tf.float32) / 255.0 - 0.5) / 0.5

    """
    GRAPY
    Changes
        Scarf(12) -> Head(1)
        Left(8), Right Leg(9) -> Lower Body (3)
        Left(10), Right Shoe(11) ->  Lower Body (3)
        Socks(14) ->  Lower Body (3)
        Lower Clothes (3) -> Lower Body (3)
    New labels
        {
            0: Background, 
            1: Head, 
            2: UpperClothes, 
            3: Lower Body, 
            4: Dresses
            5: TorsoSkin, 
            6: LeftArm, 
            7: RightArm
            8: Gloves,
        }
    """
    grapy = tf.cast(tf.image.decode_png(parsed_features["personMask"], channels=1), tf.float32)
    mapping = tf.constant([0, 1, 1, 8, 1, 2, 4, 2, 3, 3, 5, 1, 3, 1, 6, 7, 3, 3, 3, 3, 2, 2, 2])
    grapy = tf.cast(tf.gather(mapping, tf.cast(grapy, dtype=tf.int32)), tf.float32)

    # grapy = tf.cast(tf.image.decode_png(parsed_features["personMask"], channels=1), tf.float32)
    # mapping = tf.constant([0, 1, 1, 13, 1, 2, 4, 2, 14, 3, 5, 12, 3, 1, 6, 7, 8, 9, 10, 11, 2, 2, 2])
    # grapy = tf.gather(mapping, tf.cast(grapy, dtype=tf.int32))
    # grapy = tf.cast(grapy, tf.float32)


    densepose = tf.image.decode_png(parsed_features["densepose"], channels=3)
    dp_seg = tf.cast(densepose[..., 0], tf.int32)
    dp_seg = tf.cast(tf.one_hot(dp_seg, depth=25), tf.float32)
    dp_uv = (tf.cast(densepose[..., 1:], tf.float32) / 255.0 - 0.5) / 0.5

    """
    DENSEPOSE
    Changes
        L, R Foot(4, 5) + Upper Leg (6, 7) + Lower Leg (8, 9) -> Legs (4)
        Re-label upper limbs
    
    New Labels
        0      = Background
        1      = Torso
        2      = Right Hand
        3      = Left Hand
        4      = Legs
        5 = Upper Arm Left
        6 = Upper Arm Right
        7 = Lower Arm Left
        8 = Lower Arm Right
        9 = Head
    """
    densepose = tf.cast(tf.image.decode_png(parsed_features["densepose"], channels=1), tf.float32)
    mapping = tf.constant([0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 6, 7, 8, 9])
    densepose = tf.gather(mapping, tf.cast(densepose, dtype=tf.int32))
    densepose = tf.cast(densepose, tf.float32)

    dp_seg = tf.cast(densepose[..., 0], tf.int32)
    dp_seg = tf.one_hot(dp_seg, depth=10)
    densepose = tf.cast(dp_seg, tf.float32)

    person_no = parsed_features["personno"]
    is_shorts = parsed_features["is_shorts"]
    is_short_sleeve = parsed_features["is_short_sleeves"]
    category = parsed_features["category"]

    # Add background to simulate bad grapy. 
    p_sil = tf.cast(grapy > 0, tf.float32)
    p_sil = erode_grapy(p_sil)                  # Add background in the silhouette.
    grapy = grapy*p_sil
    person_with_bg = person
    person = person * p_sil                     # person segmented from background.

    # Multiplying by cloth mask to get torso
    occluded_cloth_mask = filterMask(grapy, [2, 4])
    person_cloth = person*occluded_cloth_mask

    g = tf.random.uniform([])

    shape_mask = tf.cast(filterMask(grapy, [1, 3, 8, 9, 10, 11, 12, 13, 14]), dtype=tf.float32)
    lower_body = shape_mask
    upper_clothes = tf.cast(filterMask(grapy, [2, 4]), dtype=tf.float32)
    upper_body = tf.stack([filterMask(grapy[..., 0], [i]) for i in [5, 6, 7]], axis=-1)
    exp_seg = tf.concat([filterMask(grapy, [0]), upper_clothes, upper_body, lower_body], axis=-1)
    exp_seg = B.argmax(exp_seg, axis=-1)[..., None]
    exp_seg = tf.cast(exp_seg, tf.float32)

    
    # # Find Ranodm Cloth
    # densepose = tf.image.decode_png(parsed_features["densepose"], channels=3)
    # dp_seg = tf.cast(densepose[..., 0], tf.int32)
    # dp_seg = tf.one_hot(dp_seg, depth=25)
    # dp_seg = tf.cast(dp_seg, tf.float32)
    # dp_uv = (tf.cast(densepose[..., 1:], tf.float32) / 255.0 - 0.5) / 0.5
    # densepose = tf.concat([dp_seg, dp_uv], axis=-1)

    px = tf.where(p_sil == 1)
    xmin = tf.cast(tf.reduce_min(px[:, 0]), tf.int32)
    ymin = tf.cast(tf.reduce_min(px[:, 1]), tf.int32)

    xmax = tf.cast(tf.reduce_max(px[:, 0]), tf.int32)
    ymax = tf.cast(tf.reduce_max(px[:, 1]), tf.int32)

    random_cloth = prep_cloth_random(occluded_cloth_mask, person_cloth, [person_with_bg, xmin, ymin, xmax, ymax])       # Used to add cloth part to simulate bad grapy.


    torso = filterMask(grapy, [5]) * filterMask(exp_seg, [2])                   # find complete torso skin.
    curvy_torso = curvy_cut(torso)                                              # cut some portion of torso skin.
    torso_retrained = tf.cast(curvy_torso * torso, tf.bool)                     # find the torso skin to retain.
    torso_mask = tf.cast(torso_retrained, tf.float32)

    prior_torso = tf.cast(torso_retrained, tf.float32)
    prior_torso, eroded_part_torso = simulate_misalign(prior_torso)             # erode retained torso skin.
    random_cloth2 = prep_cloth_random(occluded_cloth_mask, person_cloth, [person_with_bg, xmin, ymin, xmax, ymax])


    # Modify the function.
    left_hand = tf.cast(filterMask(grapy, [6]) * filterMask(exp_seg, [3]), dtype=tf.bool)           # find left hand mask from grapy.
    right_hand = tf.cast(filterMask(grapy, [7]) * filterMask(exp_seg, [4]), dtype=tf.bool)          # find right hand mask from grapy.

    left_palm = tf.cast(filterMask(tf.cast(B.argmax(densepose, -1), tf.float32), [2])[:, :, None], dtype=tf.bool)
    right_palm = tf.cast(filterMask(tf.cast(B.argmax(densepose, -1), tf.float32), [3])[:, :, None], dtype=tf.bool)
    
    left_hand = tf.cast(tf.math.logical_or(left_hand, left_palm), dtype=tf.float32)
    right_hand = tf.cast(tf.math.logical_or(right_hand, right_palm), dtype=tf.float32)

    curvy_left = curvy_cut(left_hand)                                           # mask to cut the left hand.
    curvy_right = curvy_cut(right_hand)                                         # mask to cut the right hand.
    left_hand_retained = tf.cast(curvy_left * left_hand, dtype=tf.bool)                  # cut curvy_left portion from left hand.
    right_hand_retained = tf.cast(curvy_right * right_hand, dtype=tf.bool)               # cut curvy_right portion from right hand.
    hand_mask = tf.math.logical_or(left_hand_retained, right_hand_retained)               # combine both left and right parts.
    hand_mask = tf.cast(hand_mask, tf.float32)

    prior_left = tf.cast(left_hand_retained, tf.float32)
    prior_right = tf.cast(right_hand_retained, tf.float32)
    
    prior_left, eroded_part_cloth_left = simulate_misalign(prior_left)                               # erodes the retained left_hand.

    prior_right, eroded_part_cloth_right = simulate_misalign(prior_right)                            # erodes the retained left_hand.


    # Model inputs 
    # Isolating the skin inpainting by giving tryon cloth as input
    person_priors_mask = filterMask(exp_seg, [1, 5]) + hand_mask + torso_mask
    person_priors_mask = tf.cast(person_priors_mask>0, tf.float32)
    person_priors = person * person_priors_mask 

    inpaint_region = filterMask(grapy, [5, 6, 7]) * filterMask(exp_seg, [2, 3, 4])

    k = tf.random.uniform([])
    use_augmentation = True

    if use_augmentation:
        if k < 0.1:
            person_priors = person_priors*(1-eroded_part_cloth_left) + eroded_part_cloth_left*random_cloth              # remove the erroded part and add random cloth in left hand.                                           # remove the erorded part from skin prior
        elif k < 0.30:
            person_priors = person_priors*(1-eroded_part_cloth_right) + eroded_part_cloth_right*random_cloth            # remove the erroded part and add random cloth in right hand.
        elif k < 0.5:
            person_priors = person_priors*(1-eroded_part_cloth_left) + eroded_part_cloth_left*random_cloth              # remove the erroded part and add random cloth in left hand.
            person_priors = person_priors*(1-eroded_part_cloth_right) + eroded_part_cloth_right*random_cloth            # remove the erroded part and add random cloth in right hand.
        else:
            pass
        
        if k>0.5:
            person_priors = person_priors*(1-eroded_part_torso) + eroded_part_torso*random_cloth2                       # remove the erroded part and add random cloth part in torso skin.
        else:
            pass
    
    # add hands.
    densepose = tf.image.decode_png(parsed_features["densepose"], channels=3)
    densepose = tf.cast(densepose[..., :1], tf.float32)
    hands = tf.cast(densepose == 2, tf.float32) + tf.cast(densepose == 3, tf.float32)
    hands = tf.cast(hands>0, tf.float32)
    person_priors = person_priors*(1-hands) + hands*person
    inpaint_region = inpaint_region*(1-hands)

    person_priors = tf.image.resize_with_pad(person_priors, 256, 256)
    inpaint_region = tf.image.resize_with_pad(inpaint_region, 256, 256)
    person = tf.image.resize_with_pad(person, 256, 256)
    exp_seg = tf.image.resize_with_pad(exp_seg, 256, 256)

    data = {
        "person_priors": person_priors,          # input to generator
        "inpaint_region": inpaint_region,       # input to generator

        "person": person,                 # ground truth
        "personno": person_no,

        "is_short_sleeve": is_short_sleeve,
        "category": category,
        "person_with_bg": person_with_bg,
        "densepose": densepose,
        "exp_seg": exp_seg,
    }

    model_inputs = {
        "person_priors": person_priors,          # input to generator
        "inpaint_region": inpaint_region,       # input to generator
        "exp_seg": exp_seg,
    }

    return data, model_inputs


def create_dataset(parse_func, filter_func, tfrecord_path, batch_size, mode, data_split, device, debug):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(
        parse_func,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.filter(filter_func)

    if debug:
        num_lines = 5000
        batch_size = 1
    else:
        # num_lines = sum(1 for _ in dataset)
        num_lines = 15000

    if mode == "train":
        dataset = dataset.take(int(data_split * num_lines))
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)
        num_data = int(data_split * num_lines)

    elif mode == "val":
        dataset = dataset.skip(int(data_split * num_lines))
        num_data = int((1-data_split) * num_lines)

    elif mode == "k_worst":
        dataset = dataset.take(data_split * num_lines)
        num_data = int(data_split * num_lines)

    if mode != "k_worst":
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    if device != "colab_tpu":
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, num_data
    else:
        return dataset

def define_dataset(tfrecord_path, batch_size, train=True, test=False):
    per_replica_train_batch_size = batch_size
    per_replica_val_batch_size = batch_size
    if test:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            batch_size=per_replica_train_batch_size,
            mode="k_worst",
            data_split=1,
            device='gpu',
            debug=False
        )
        return data_gen, dataset_length

    if train:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            batch_size=per_replica_train_batch_size,
            mode="train",
            data_split=0.8,
            device='gpu',
            debug=False
        )
    else:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            batch_size=per_replica_val_batch_size,
            mode="val",
            data_split=0.8,
            device='gpu',
            debug=False
        )
    return data_gen, dataset_length