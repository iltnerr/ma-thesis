def NMS_keynet(score_map, size):

    score_map = score_map * (score_map == maximum_filter(score_map, footprint=np.ones((size, size))))

    return score_map

def NMS(prob, size, iou=0.1, min_prob=0.01):
    """
    See https://github.com/rpautrat/SuperPoint

    Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field).
    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the bouding boxes.
        iou: a scalar, the IoU overlap threshold.
        min_prob: a threshold under which all probabilities are discarded before NMS.
    """
    with tf.name_scope('box_nms'):

        prob_unstack = tf.unstack(prob, num=CONFIG["BATCH_SIZE"])
        prob_stack = []

        for prob in prob_unstack:

            # heatmap shape [H, W]
            prob = tf.squeeze(prob)

            # NMS
            pts = tf.cast(tf.where(tf.greater_equal(prob, min_prob)), tf.float32)
            # size = tf.constant(size/2.)
            size = size/2.
            boxes = tf.concat([pts-size, pts+size], axis=1)
            scores = tf.gather_nd(prob, tf.cast(pts, tf.int32))

            # try gpu execution
            indices = tf.image.non_max_suppression(boxes, scores, tf.shape(boxes)[0], iou)

            """
            with tf.device('/cpu:0'):  # there was a bug with gpu execution. may have been fixed by now.
                indices = tf.image.non_max_suppression(
                        boxes, scores, tf.shape(boxes)[0], iou)
            """

            pts = tf.gather(pts, indices)
            scores = tf.gather(scores, indices)
            prob = tf.scatter_nd(tf.cast(pts, tf.int32), scores, tf.shape(prob))

            # stack
            prob = tf.expand_dims(prob, axis=-1)
            prob_stack.append(prob)

        prob_nms = tf.stack(prob_stack)

    return prob_nms