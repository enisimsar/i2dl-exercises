"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    scores = X.dot(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # Softmax Loss
    for i in range(num_train):
        f = scores[i] - np.max(scores[i]) # avoid numerical instability
        softmax = np.exp(f)/np.sum(np.exp(f))
        loss += -np.log(softmax[y[i]])
        # Weight Gradients
        for j in range(num_classes):
            if j != y[i]:
                dW[:,j] += X[i] * softmax[j]
        dW[:,y[i]] -= X[i]

    # Average
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W 
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    num_train = X.shape[0]
    scores = X.dot(W)

    # Normalize the scores to avoid computational problems with the exponential
    # Normalize with max as zero
    scores = scores - np.max(scores, axis = 1).reshape(-1,1)
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    loss = -np.sum(np.log(probs[np.arange(num_train),y]))

    # Divide the loss by the number of trainig examples
    loss /= num_train
    # Add regularization
    loss += 0.5 * reg * np.sum(W * W)

    dprobs = probs
    dprobs[np.arange(num_train),y] -= 1
    dW = X.T.dot(dprobs)
    dW /= num_train

    # Gradient regularization
    dW += reg*W

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    learning_rates = [3e-7, 3e-6]
    regularization_strengths = [1e3, 2e5]
    
    def get_random_params(lr_stats, reg_stats):
        lr = np.random.uniform(lr_stats[0], lr_stats[1])
        reg = np.random.uniform(reg_stats[0], reg_stats[1])
        return lr, reg
    
    num_experiment = 20
    for i in range(num_experiment):
        print(f"Trying {i+1:02d}/{num_experiment} subset...", end=" ")
        lr, rg = get_random_params(learning_rates, regularization_strengths)
        softmax = SoftmaxClassifier()
        softmax.train(X_train, y_train, learning_rate=lr, reg=rg, num_iters=1500)

        y_train_pred = softmax.predict(X_train)
        train_accuracy = np.mean(y_train_pred == y_train)

        y_val_pred = softmax.predict(X_val)
        val_accuracy = np.mean(y_val_pred == y_val)

        results[(lr,rg)] = (train_accuracy, val_accuracy)
        if best_val < val_accuracy:
            best_val = val_accuracy
            best_softmax = softmax
    
        all_classifiers.append((softmax, val_accuracy))
        print("Val Accuracy:", val_accuracy)

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
