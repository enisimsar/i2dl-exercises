import numpy as np
import torch


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        #######################################################################
        # TODO:                                                               #
        # Write your own personal training method for our solver. In each     #
        # epoch iter_per_epoch shuffled training batches are processed. The   #
        # loss for each batch is stored in self.train_loss_history. Every     #
        # log_nth iteration the loss is logged. After one epoch the training  #
        # accuracy of the last mini batch is logged and stored in             #
        # self.train_acc_history. We validate at the end of each epoch, log   #
        # the result and store the accuracy of the entire validation set in   #
        # self.val_acc_history.                                               #
        #                                                                     #
        # Your logging could like something like:                             #
        #   ...                                                               #
        #   [Iteration 700/4800] TRAIN loss: 1.452                            #
        #   [Iteration 800/4800] TRAIN loss: 1.409                            #
        #   [Iteration 900/4800] TRAIN loss: 1.374                            #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                           #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                           #
        #   ...                                                               #
        #######################################################################
        for epoch in range(num_epochs):
            model.train()

            train_scores = []
            train_losses = []
            for i, (batch, target) in enumerate(train_loader, 1):
                batch = batch.to(device)
                target = target.to(device)

                optim.zero_grad()
                outputs = model(batch)
                loss = self.loss_func(outputs, target)
                train_losses.append(loss.data.cpu().numpy())
                loss.backward()
                optim.step()

                _, preds = torch.max(outputs, 1)

                targets_mask = target >= 0
                scores = np.mean((preds == target)[targets_mask].data.cpu().numpy())
                train_scores.append(scores)

                if log_nth and i % log_nth == 0:
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         np.mean(train_losses)))
            
            self.train_loss_history.append(np.mean(train_losses))
            train_acc = np.mean(train_scores)
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   np.mean(train_losses)))

            model.eval()

            val_losses = []
            val_scores = []
            with torch.no_grad():
                for batch, target in val_loader:
                    batch = batch.to(device)
                    target = target.to(device)

                    outputs = model(batch)
                    loss = self.loss_func(outputs, target)
                    val_losses.append(loss.data.cpu().numpy())

                    _, preds = torch.max(outputs, 1)

                    targets_mask = target >= 0
                    scores = np.mean((preds == target)[targets_mask].data.cpu().numpy())
                    val_scores.append(scores)

            val_acc, val_loss = np.mean(val_scores), np.mean(val_losses)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_loss))
        

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        print('FINISH.')
