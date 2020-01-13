import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################

        layers = []
        conv2d = nn.Conv2d(1, 16, kernel_size=4)
        layers += [
            conv2d, 
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3)
        ]

        conv2d = nn.Conv2d(16, 32, kernel_size=3)
        layers += [
            conv2d, 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3)
        ]

        conv2d = nn.Conv2d(32, 64, kernel_size=3)
        layers += [
            conv2d, 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3)
        ]

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, 30),
        )

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
