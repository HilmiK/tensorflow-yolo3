# -*- coding:utf-8 -*-
# author: gaochen
# date: 2018.06.04


import numpy as np
import tensorflow as tf
import os

class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):

        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()


    def _get_class(self):

        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):

        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)



    def _batch_normalization_layer(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):

        bn_layer = tf.layers.batch_normalization(inputs = input_layer,
            momentum = norm_decay, epsilon = norm_epsilon, center = True,
            scale = True, training = training, name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)


    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):

        if strides > 1:
            inputs = tf.pad(inputs, paddings = [[0, 0], [1, 0], [1, 0], [0, 0]], mode = 'CONSTANT')
        conv = tf.layers.conv2d(
            inputs = inputs, filters = filters_num,
            kernel_size = kernel_size, strides = [strides, strides], kernel_initializer = tf.glorot_uniform_initializer(),
            padding = ('SAME' if strides == 1 else 'VALID'), kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4), use_bias = use_bias, name = name)
        return conv


    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):

        layer = self._conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = "conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index


    def _darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):

        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index,  filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        return  route1, route2, conv, conv_index


    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):

        conv = self._conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index


    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):

        conv_index = 1
        conv2d_26, conv2d_45, conv, conv_index = self._darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_45], axis = -1, name = 'route_0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training,norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = 'route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]



    def yolo_head(self, feats, anchors, num_classes, input_shape, training = True):

        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype = tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)
        grid = tf.cast(grid, tf.float32)
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        if training == True:
            return grid, predictions, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs


    def yolo_boxes_scores(self, feats, anchors, num_classes, input_shape, image_shape):
 
        input_shape = tf.cast(input_shape, tf.float32)
        image_shape = tf.cast(image_shape, tf.float32)
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape, training = False)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.
        box_max = box_yx + box_hw / 2.
        boxes = tf.concat(
            [box_min[..., 0:1],
             box_min[..., 1:2],
             box_max[..., 0:1],
             box_max[..., 1:2]],
            axis = -1
        )
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        boxes = tf.reshape(boxes, [-1, 4])
        boxes_scores = box_confidence * box_class_probs
        boxes_scores = tf.reshape(boxes_scores, [-1, num_classes])
        return boxes, boxes_scores


    def box_iou(self, box1, box2):
  
        box1 = tf.expand_dims(box1, -2)
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.
        box1_maxs = box1_xy + box1_wh / 2.

        box2 = tf.expand_dims(box2, 0)
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.
        box2_maxs = box2_xy + box2_wh / 2.

        intersect_mins = tf.maximum(box1_mins, box2_mins)
        intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou



    def yolo_loss(self, yolo_output, y_true, anchors, num_classes, ignore_thresh = .5):
 
        loss = 0
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = [416.0, 416.0]
        grid_shapes = [tf.cast(tf.shape(yolo_output[l])[1:3], tf.float32) for l in range(3)]
        for index in range(3):
            object_mask = y_true[index][..., 4:5]
            class_probs = y_true[index][..., 5:]
            grid, predictions, pred_xy, pred_wh = self.yolo_head(yolo_output[index], anchors[anchor_mask[index]], num_classes, input_shape, training = True)

            pred_box = tf.concat([pred_xy, pred_wh], axis = -1)
            raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid
            object_mask_bool = tf.cast(object_mask, dtype = tf.bool)
            raw_true_wh = tf.log(tf.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1], 0), tf.ones_like(y_true[index][..., 2:4]), y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1]))

            box_loss_scale = 2 - y_true[index][..., 2:3] * y_true[index][..., 3:4]
            ignore_mask = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)
            def loop_body(internal_index, ignore_mask):
                true_box = tf.boolean_mask(y_true[index][internal_index, ..., 0:4], object_mask_bool[internal_index, ..., 0])
                iou = self.box_iou(pred_box[internal_index], true_box)

                best_iou = tf.reduce_max(iou, axis = -1)
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
                return internal_index + 1, ignore_mask
            _, ignore_mask = tf.while_loop(lambda internal_index, ignore_mask : internal_index < tf.shape(yolo_output[0])[0], loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis = -1)
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels = raw_true_xy, logits = predictions[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) * ignore_mask
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels =  class_probs, logits = predictions[..., 5:])
            xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            class_loss = tf.reduce_sum(class_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)

            loss += xy_loss + wh_loss + confidence_loss + class_loss

        return loss
