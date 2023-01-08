import keras.backend as K
import numpy as np
#from utils import  get_logging
#logger = get_logging()

import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

SMOOTH = tf.cast(1e-12, tf.float32)

#---------------------------------Deprecated--------------------------------------------
def competitionMetric2(y_true, y_pred): #any shape can go - can't be a loss function
    def castB(x):
        return K.cast(x, bool)
    def castF(x):
        return K.cast(x, K.floatx())


    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)
    y_pred = castF(K.greater(y_pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(y_true, axis=-1)
    predSum = K.sum(y_pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(y_true, truePositiveMask)
    testPred = tf.boolean_mask(y_pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred)
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives)

    return (truePositives + trueNegatives) / castF(K.shape(y_true)[0])

def iou_loss_core(y_true, y_pred):  #this can be used as a loss if you make it negative
    intersection = y_true * y_pred
    notTrue = 1 - y_true
    union = y_true + (notTrue * y_pred)

    return (K.sum(intersection, axis=-1) + K.epsilon()) / (K.sum(union, axis=-1) + K.epsilon())

def get_iou_score(class_weights=1., smooth=SMOOTH, per_image=True):
    def iou_score(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
        r""" The `Jaccard index`_, also known as Intersection over Union and the Jaccard similarity coefficient
        (originally coined coefficient de communauté by Paul Jaccard), is a statistic used for comparing the
        similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets,
        and is defined as the size of the intersection divided by the size of the union of the sample sets:

        .. math:: J(A, B) = \frac{A \cap B}{A \cup B}

        Args:
            y_true: ground truth 4D keras tensor (B, H, W, C)
            y_pred: y_predediction 4D keras tensor (B, H, W, C)
            class_weights: 1. or list of class weights, len(weights) = C
            smooth: value to avoid division by zero
            per_image: if ``True``, metric is calculated as mean over images in batch (B),
                else over whole batch

        Returns:
            IoU/Jaccard score in range [0, 1]

        .. _`Jaccard index`: https://en.wikipedia.org/wiki/Jaccard_index

        """
        if per_image:
            axes = [1, 2]
        else:
            axes = [0, 1, 2]

        intersection = K.sum(y_true * y_pred, axis=axes)
        union = K.sum(y_true + y_pred, axis=axes) - intersection
        iou = (intersection + smooth) / (union + smooth)

        # mean per image
        if per_image:
            iou = K.mean(iou, axis=0)

        # weighted mean per class
        iou = K.mean(iou * class_weights)

        return iou

    """Change default parameters of IoU/Jaccard score

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        ``callable``: IoU/Jaccard score
    """
    def score(y_true, y_pred):
        return iou_score(y_true, y_pred, class_weights=class_weights, smooth=smooth, per_image=per_image)

    return score

def get_f_score(class_weights=1, beta=1, smooth=SMOOTH, per_image=True):
    """Change default parameters of F-score score

    Args:
        class_weights: 1. or list of class weights, len(weights) = C
        smooth: value to avoid division by zero
        beta: f-score coefficient
        per_image: if ``True``, metric is calculated as mean over images in batch (B),
            else over whole batch

    Returns:
        ``callable``: F-score
    """

    def f_score(y_true, y_pred, class_weights=1, beta=1, smooth=SMOOTH,
                per_image=True):
        r"""The F-score (Dice coefficient) can be intery_predeted as a weighted average of the y_predecision and recall,
        where an F-score reaches its best value at 1 and worst score at 0.
        The relative contribution of ``y_predecision`` and ``recall`` to the F1-score are equal.
        The formula for the F score is:

        .. math:: F_\beta(y_predecision, recall) = (1 + \beta^2) \frac{y_predecision \cdot recall}
            {\beta^2 \cdot y_predecision + recall}

        The formula in terms of *Type I* and *Type II* errors:

        .. math:: F_\beta(A, B) = \frac{(1 + \beta^2) TP} {(1 + \beta^2) TP + \beta^2 FN + FP}


        where:
            TP - true positive;
            FP - false positive;
            FN - false negative;

        Args:
            y_true: ground truth 4D keras tensor (B, H, W, C)
            y_pred: y_predediction 4D keras tensor (B, H, W, C)
            class_weights: 1. or list of class weights, len(weights) = C
            beta: f-score coefficient
            smooth: value to avoid division by zero
            per_image: if ``True``, metric is calculated as mean over images in batch (B),
                else over whole batch

        Returns:
            F-score in range [0, 1]

        """
        if per_image:
            axes = [1, 2]
        else:
            axes = [0, 1, 2]

        tp = K.sum(y_true * y_pred, axis=axes)
        fp = K.sum(y_pred, axis=axes) - tp
        fn = K.sum(y_true, axis=axes) - tp

        score = ((1 + beta ** 2) * tp + smooth) \
                / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

        # mean per image
        if per_image:
            score = K.mean(score, axis=0)

        # weighted mean per class
        score = K.mean(score * class_weights)

        return score
    def score(y_true, y_pred):
        return f_score(y_true, y_pred, class_weights=class_weights, beta=beta, smooth=smooth, per_image=per_image)

    return score

#-------------------------------------------------------------------------------------


def mean_iou(y_true, y_pred, ignore_label = (0)):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        ignore_label: ignore label index
    Returns:
        the scalar IoU value (mean over all labels)
    """
    def iou(y_true, y_pred, label: int):
        """
        Return the Intersection over Union (IoU) for a given label.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
            label: the label to return the IoU for
        Returns:
            the IoU for the given label
        """
        # extract the label values using the argmax operator then
        # calculate equality of the predictions and truths to the label
        y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
        y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
        # calculate the |intersection| (AND) of the labels
        intersection = K.sum(y_true * y_pred)
        # calculate the |union| (OR) of the labels
        union = K.sum(y_true) + K.sum(y_pred) - intersection
        # avoid divide by zero - if the union is zero, return 1
        # otherwise, return the intersection over union
        return K.switch(K.equal(union, 0), 1.0, intersection / union)

    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        if label not in [ignore_label]:
            total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU

    if ignore_label:
        return total_iou / (num_labels - len(ignore_label))
    else:
        return total_iou / num_labels


def Mean_IOU(y_true, y_pred):
    '''
    calculate for pascal voc
    :param y_true:
    :param y_pred:
    :return:
    '''
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.cast(true_labels & pred_labels,tf.int32)
        union = tf.cast(true_labels | pred_labels,tf.int32)
        legal_batches = K.sum(tf.cast(true_labels,tf.int32), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


def pixel_accuracy(y_true, y_pred, is_keras = True, ignore_bg = False):
    '''
    ignore label = 0, which is background
    :param y_true:
    :param y_pred: probability label or argmax label
    :param is_keras:
    :param ignore_bg: ignore count background or not
    :return:
    '''
    if not is_keras:
        y_pred = np.asarray(y_pred)
        y_pred = np.argmax(y_pred, -1)
        y_true = np.asarray(y_true)
        y_true = np.argmax(y_true, -1)
    else:
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.argmax(y_true, axis=-1)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    # pixel label maybe all belongs to background
    if ignore_bg:
        pixel_labeled = np.sum(y_true > 0) if not is_keras else K.sum(tf.cast(y_true > 0, tf.float32))
    else:
        pixel_labeled = np.sum(y_true >= 0) if not is_keras else K.sum(tf.cast(y_true >= 0, tf.float32))

    pixel_correct = np.sum((y_pred == y_true) * (y_true > 0)) if not is_keras else K.sum(tf.cast(K.equal(y_true, y_pred), tf.float32))

    pixel_accuracy = 1.0 * pixel_correct / (pixel_labeled)

    return pixel_accuracy
    # return pixel_correct, pixel_labeled

def tf_miou(y_true, y_pred, num_classes = 4):
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    mean_iou, update_op = tf.metrics.mean_iou(labels = y_true, predictions =
    y_pred,
                                              num_classes = num_classes, name =
                                              'mean_iou')

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'mean_iou' in
                   i.name.split(
                       '/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    with tf.control_dependencies([update_op]):
        return tf.identity(mean_iou)


def tf_pred_miou(y_true, y_pred, num_classes = 4):
    with tf.Session() as sess:
        y_pred =  tf.constant(y_pred)
        y_true = tf.constant(y_true)
        iou,conf_mat = tf.metrics.mean_iou(y_true, y_pred, num_classes=num_classes)
        sess.run(tf.local_variables_initializer())
        sess.run([conf_mat])
        iou = sess.run([iou])[0]
        return iou



def all_scores(y_true, y_pred, nb_classes, ignore_label = None, return_score = ('mIOU', 'pa')):

    '''
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> a = np.array([ np.random.randint(0, 4) for i in range(25)])
    >>> b = np.array([ np.random.randint(0, 4) for i in range(25)])
    >>> c = np.copy(b)
    >>> c[0] = 1
    >>> print(f'{a}\n{b}\n{c}')
    >>> b.shape = -1
    >>> c.shape = -1
    >>> print(all_scores(a, b, nb_classes=4, ignore_label=None, return_score=('pa', 'mA', 'cIOU', 'fW_IOU', 'mIOU')))

    :param label_trues: [h, w, c] if is_tf else [c, h, w]
    :param label_preds: [h, w, c] if is_tf else [c, h, w]
    :param n_class:
    :param ignore_label:  list which label will not calculate, default is 0
    :param return_score: ('pa', 'mA', 'cIOU', 'fW_IOU', 'mIOU', 'confusion', 'classification_report')
    :return:
    '''
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    hist = confusion_matrix(y_true = y_true, y_pred=y_pred, labels=[i for i in range(nb_classes)])

    if ignore_label:
        ignore_label = ignore_label if type(ignore_label) == list else [ignore_label]
        for i in [ignore_label]:
            hist[i, :] = 0
            hist[:, i] = 0

    hist[np.isnan(hist)] = 0
    #logger.debug(hist.flatten())

    scores = {}

    if 'classification_report' in return_score:
        scores['classification_report'] = classification_report(y_true, y_pred, labels=[i for i in range(nb_classes)], target_names=['%02d'%i for i in range(nb_classes)])

    if 'confusion' in return_score:
        scores['confusion'] = hist

    if 'pa' in return_score:
        acc = np.diag(hist).sum() / hist.sum()
        scores['pa'] = acc

    if 'mPA' in return_score:
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + np.spacing(1))
        acc_cls = np.nanmean(acc_cls)
        scores['mPA'] = acc_cls

    if 'cIOU' in return_score or 'mIOU' in return_score or 'fW_IOU' in return_score:
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + np.spacing(1))

        if 'mIOU' in return_score:
            valid = hist.sum(axis=1) > 0  # added
            mean_iu = np.nanmean(iu[valid])
            scores['mIOU'] = mean_iu

        if 'cIOU' in return_score:
            cls_iu = dict(zip(range(nb_classes), iu))
            scores['cIOU'] = cls_iu

        if 'fW_IOU' in return_score:
            freq = hist.sum(axis=1) / hist.sum()
            fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

            scores['fW_IOU'] = fwavacc

    return scores

if __name__ == '__main__':
    import numpy as np
    # np.random.seed(1)
    a = np.array([ np.random.randint(0, 4) for i in range(25)])
    b = np.array([ np.random.randint(0, 4) for i in range(25)])
    c = np.copy(b)
    c[0] = 1
    print(f'{a}\n{b}\n{c}')
    b.shape = -1
    c.shape = -1


    print(all_scores(a, b, nb_classes=4, ignore_label=None, return_score=('pa', 'mA', 'cIOU', 'fW_IOU', 'mIOU')))


    # d = confusion_matrix(b, c, labels=[ 2, 3])
    # print(d)
