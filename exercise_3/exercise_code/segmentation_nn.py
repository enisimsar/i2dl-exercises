"""SegmentationNN"""
import torch
import torch.nn as nn


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        import torchvision.models as models
        
        self.n_class = num_classes
        
        self.features = models.mobilenet_v2(pretrained=True).features
        
        self.fcn = nn.Sequential(
            nn.Conv2d(1280, 1024, 8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024, 2048, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(2048, num_classes, 1)
        )

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        import torch.nn.functional as F

        x_input = x
        x = self.features(x)
        x = self.fcn(x)
        x = F.interpolate(x, x_input.size()[2:], mode='bilinear', align_corners=True).contiguous()
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
