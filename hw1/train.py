import tqdm
import torch
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score
import json

from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
)


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
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

        # Perform one-hot encoding on all token Tensors
        # Specify num_classes to preserve size of array across data inputs
        data_dict['instructions'] = F.one_hot(data_dict['instructions'],
                                              num_classes = args.vocab_size)
        data_dict['actions'] = F.one_hot(data_dict['actions'],
                                         num_classes = len(actions_to_index))
        data_dict['targets'] = F.one_hot(data_dict['targets'],
                                         num_classes = len(targets_to_index))
    
    print('tokenized')
    
    train_loader = torch.utils.data.DataLoader(train_dict)
    val_loader = torch.utils.data.DataLoader(val_dict)
    return train_loader, val_loader, maps


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    model = None
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
    action_criterion = None
    target_criterion = None
    optimizer = None

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
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in loader:
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(inputs, labels)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

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

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.cpu().numpy())
        target_preds.extend(target_preds_.cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

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

    for epoch in tqdm.tqdm(range(args.num_epochs)):

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
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

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

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

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
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=5, help="number of epochs between every eval loop"
    )
    parser.add_argument("--vocab_size", default=1000, help="number of tokens in vocab")

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    args = parser.parse_args()

    main(args)
