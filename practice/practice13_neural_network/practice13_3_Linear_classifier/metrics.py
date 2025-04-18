def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp = prediction[(ground_truth == True) & (prediction == True)].shape[0]
    tn = prediction[(ground_truth == False) & (prediction == False)].shape[0]
    fp = prediction[(ground_truth == False) & (prediction == True)].shape[0]
    fn = prediction[(ground_truth == True) & (prediction == False)].shape[0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    return prediction[ground_truth == prediction].shape[0] / prediction.shape[0]
