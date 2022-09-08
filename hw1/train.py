from tqdm import tqdm
import torch
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import json
import torchviz

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)

def preprocess_data(args):
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #
    # Load the data from the json
    data = json.load(open(args.in_data_fn))

    # Use the utils to construct the necessary maps
    vocab_to_index, index_to_vocab, max_seq_len = (
        build_tokenizer_table(data['train'], vocab_size = args.vocab_size)
    )
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = (
        build_output_tables(data['train'])
    )

    maps = [vocab_to_index, index_to_vocab, actions_to_index, 
            index_to_actions, targets_to_index, index_to_targets]

    # List of tokenized instructions, actions, and targets for each dataset
    train_dict = {"instructions": [], "actions": [], "targets": []}
    val_dict = {"instructions": [], "actions": [], "targets": []}
    data_dicts = [train_dict, val_dict]

    print('begin')

    # Iterate through every episode in each dataset
    for dataset, data_dict in zip(data.values(), data_dicts):
        for episode in dataset:
            for inst, outseq in episode:
                # Tokenize and store instructions
                # Begin each instruction with a <start> token
                inst_tokens = [vocab_to_index['<start>']]
                inst = preprocess_string(inst)

                # Word level tokenization
                for word in inst.split(" "):
                    word_idx = vocab_to_index.get(word, vocab_to_index['<unk>'])
                    inst_tokens.append(word_idx)
                
                # Truncate instruction tokens to max_seq_len
                if len(inst_tokens) > max_seq_len:
                    inst_tokens = inst_tokens[:max_seq_len]
                
                # Add <pad> and <end> tokens where applicable
                if len(inst_tokens) < max_seq_len:
                    for _ in range(max_seq_len - len(inst_tokens) - 1):
                        inst_tokens.append(vocab_to_index['<pad>'])
                    inst_tokens.append(vocab_to_index['<end>'])

                # Tokenize and store action and target
                a, t = outseq

                # Store tokens in relevant data dict
                data_dict['instructions'].append(inst_tokens)
                data_dict['actions'].append(actions_to_index[a])
                data_dict['targets'].append(targets_to_index[t])  

        # Convert all token lists to Tensors of int64
        data_dict['instructions'] = torch.Tensor(data_dict['instructions']).to(torch.int64)
        data_dict['actions'] = torch.Tensor(data_dict['actions']).to(torch.int64)
        data_dict['targets'] = torch.Tensor(data_dict['targets']).to(torch.int64)

        # # Perform one-hot encoding on all token Tensors
        # # Specify num_classes to preserve size of array across data inputs
        # data_dict['instructions'] = F.one_hot(data_dict['instructions'],
        #                                       num_classes = args.vocab_size)
        data_dict['actions'] = F.one_hot(data_dict['actions'],
                                         num_classes = len(actions_to_index))
        data_dict['targets'] = F.one_hot(data_dict['targets'],
                                         num_classes = len(targets_to_index))
    
    print('preprocessed')

    return train_dict, val_dict, maps

class alfred_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        """
        Initialize the Dataset, which has the same structure as
        the data_dict provided by preprocess_data(), but implements
        the necessary __len__() and __getitem__() methods. 
        """
        super().__init__()
        self.instructions = data_dict['instructions']
        self.actions = data_dict['actions']
        self.targets = data_dict['targets']
    
    def __len__(self):
        """
        Verify that all attributes are of the same length,
        then return that length. 
        """
        assert len(self.instructions) == len(self.actions)
        assert len(self.actions) == len(self.targets)
        return len(self.instructions)

    def __getitem__(self, idx):
        """
        Returns a 3-tuple of the instructions, actions,
        and targets at the idx. 
        """
        return self.instructions[idx], self.actions[idx], self.targets[idx]

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # Process the text into tokens stored in dictionaries
    train_dict, val_dict, maps = preprocess_data(args)

    # Store the tokenized data in a custom defined Dataset object
    train_dataset = alfred_dataset(train_dict)
    val_dataset = alfred_dataset(val_dict)

    # Wrap the Dataset objects in iterable DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    return train_loader, val_loader, maps

class LSTM(torch.nn.Module):
    def __init__(self, args, maps):
        """
        Defines our custom LSTM class. 
        """
        # The internet says I should initialize my parent class?
        # This seems silly, shouldn't Python take care of that?
        # Not sure, including it anyways. See here: 
        # https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
        super().__init__()

        # Initialize some hyperparameters
        self.embedding_dim = 128
        self.lstm_dim = 128
        if args.maxpool:
            self.lstm_dim = int(self.lstm_dim / 2)
            self.final_hidden_dim = self.lstm_dim * 2
        else:
            self.final_hidden_dim = self.lstm_dim
        # TODO: Add feature sizing for maxpool
        self.lstm_layers = 1
        self.dropout = 0
        self.bidirectional = False
        self.action_classes = len(maps[2])
        self.target_classes = len(maps[4])

        # Initialize an embedding layer the size of our vocabulary
        self.embed = torch.nn.Embedding(num_embeddings=args.vocab_size, 
                                        embedding_dim=self.embedding_dim)

        # Initialize an LSTM block using our hyperparameters
        self.lstm = torch.nn.LSTM(input_size=self.embedding_dim, 
                                  hidden_size=self.lstm_dim,
                                  num_layers=self.lstm_layers,
                                  dropout=self.dropout,
                                  bidirectional=self.bidirectional,
                                  batch_first=True)

        # Initialize a fully-connected linear layer
        self.fc_action = torch.nn.Linear(in_features=self.final_hidden_dim,
                                         out_features=self.action_classes)
        self.fc_target = torch.nn.Linear(in_features=self.final_hidden_dim,
                                         out_features=self.target_classes)

    def forward(self, inputs):
        """
        Performs a forward pass through the LSTM. 
        """
        # Embed the inputs
        embeds = self.embed(inputs)

        # Run the embeddings through the LSTM
        lstm_output, (h_n, c_n)  = self.lstm(embeds)

        # Maxpool the LSTM word embeddings and concatenate with the hidden state
        if args.maxpool:
            max_lstm_output = F.max_pool1d(lstm_output.transpose(-1, -2), 
                                        kernel_size=lstm_output.size()[1])
            h_n = torch.cat((max_lstm_output.squeeze(), h_n.squeeze()), dim=-1)

        # Use two separate fully connected layers to predict actions and targets
        action_mass = self.fc_action(h_n).squeeze()
        target_mass = self.fc_target(h_n).squeeze()

        return action_mass, target_mass
        

def setup_model(args, maps, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #

    model = LSTM(args, maps)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    return action_criterion, target_criterion, optimizer

def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    for inputs, actions, targets in tqdm(loader):
        # put model inputs to device
        inputs, actions, targets = inputs.to(device), actions.to(device), targets.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs)

        # calculate the action and target prediction loss
        action_loss = action_criterion(actions_out.squeeze(), actions.float())
        target_loss = target_criterion(targets_out.squeeze(), targets.float())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)
        actions_ = actions.argmax(-1)
        targets_ = targets.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(actions_.cpu().numpy())
        target_labels.extend(targets_.cpu().numpy())

    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():

        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()
    model.to(device)

    # Setup lists to track loss and accuracy over epochs
    train_action_accs, train_action_losses = [],[]
    train_target_accs, train_target_losses = [],[]
    val_action_accs, val_action_losses = [],[]
    val_target_accs, val_target_losses = [],[]
    train_epoch_nums, val_epoch_nums = [],[]

    for epoch in tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(f"train action loss : {train_action_loss}")
        print(f"train target loss: {train_target_loss}")
        print(f"train action acc : {train_action_acc}")
        print(f"train target acc: {train_target_acc}")
        train_action_accs.append(train_action_acc)
        train_action_losses.append(train_action_loss)
        train_target_accs.append(train_target_acc)
        train_target_losses.append(train_target_loss)
        train_epoch_nums.append(epoch)


        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target losaccs: {val_target_acc}"
            )

            val_action_accs.append(val_action_acc)
            val_action_losses.append(val_action_loss)
            val_target_accs.append(val_target_acc)
            val_target_losses.append(val_target_loss)
            val_epoch_nums.append(epoch)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # four plots, each with a train and val line. 
    # accuracy, loss * action, target
    # ===================================================== #

    # Divides Loss by Len to find Average Loss Per Example
    train_action_losses = [i/len(loaders['train']) for i in train_action_losses]
    train_target_losses = [i/len(loaders['train']) for i in train_target_losses]
    val_action_losses = [i/len(loaders['val']) for i in val_action_losses]
    val_target_losses = [i/len(loaders['val']) for i in val_target_losses]

    # Creates four subplots on the same figure, with two lines on each plot
    # The four plots represent accuracy and loss on actions and targets
    # The two lines represent training vs validation measures
    # Credit to: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(train_epoch_nums, train_action_accs, "b--", label="train") 
    axs[0, 0].plot(val_epoch_nums, val_action_accs, "b-", label="validation")
    axs[0, 0].legend(loc="upper left")
    axs[0, 0].set_title('Action Accuracy')
    axs[0, 1].plot(train_epoch_nums, train_target_accs, "o--", label="train")
    axs[0, 1].plot(val_epoch_nums, val_target_accs, "o-", label="validation")
    axs[0, 1].legend(loc="upper left")
    axs[0, 1].set_title('Target Accuracy')
    axs[1, 0].plot(train_epoch_nums, train_action_losses, "m--", label="train")
    axs[1, 0].plot(val_epoch_nums, val_action_losses, "m-", label="validation")
    axs[1, 0].legend(loc="upper right")
    axs[1, 0].set_title('Action Loss')
    axs[1, 1].plot(train_epoch_nums, train_target_losses, "g--", label="train")
    axs[1, 1].plot(val_epoch_nums, val_target_losses, "g-", label="validation")
    axs[1, 1].legend(loc="upper right")
    axs[1, 1].set_title('Target Loss')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
        ax.xlabel('Epoch #')
    
    plt.show()


def main(args):
    device = get_device(args.force_cpu)
    torch.manual_seed(42)

    # get dataloaders
    train_loader, val_loader, maps = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, maps, device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--maxpool", action="store_true", 
                        help="model uses lstm hidden state and token outputs")
    parser.add_argument(
        "--num_epochs", default=1000, help="number of training epochs", type=int
    )
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop", type=int
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument("--vocab_size", default=1000, help="number of tokens in vocab")

    args = parser.parse_args()

    main(args)
