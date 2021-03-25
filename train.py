"""
Speech Command Recognition with torchaudio
******************************************

This tutorial will show you how to correctly format an audio dataset and
then train/test an audio classifier network on the dataset.

Colab has GPU option available. In the menu tabs, select “Runtime” then
“Change runtime type”. In the pop-up that follows, you can choose GPU.
After the change, your runtime should automatically restart (which means
information from executed cells disappear).

First, let’s import the common torch packages such as
`torchaudio <https://github.com/pytorch/audio>`__ that can be installed
by following the instructions on the website.

"""

# Uncomment the following line to run in Google Colab

# CPU:
# !pip install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# GPU:
# !pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# For interactive demo at the end:
# !pip install pydub

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchaudio

import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

import data
import model

if __name__ == "__main__":
    save_path = "runs/test_{}".format(int(time.time()))


    ######################################################################
    # Let’s check if a CUDA GPU is available and select our device. Running
    # the network on a GPU will greatly decrease the training/testing runtime.
    #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(device)


    ######################################################################
    # Importing the Dataset
    # ---------------------
    #
    # We use torchaudio to download and represent the dataset. Here we use
    # `SpeechCommands <https://arxiv.org/abs/1804.03209>`__, which is a
    # datasets of 35 commands spoken by different people. The dataset
    # ``SPEECHCOMMANDS`` is a ``torch.utils.data.Dataset`` version of the
    # dataset. In this dataset, all audio files are about 1 second long (and
    # so about 16000 time frames long).
    #
    # The actual loading and formatting steps happen when a data point is
    # being accessed, and torchaudio takes care of converting the audio files
    # to tensors. If one wants to load an audio file directly instead,
    # ``torchaudio.load()`` can be used. It returns a tuple containing the
    # newly created tensor along with the sampling frequency of the audio file
    # (16kHz for SpeechCommands).
    #
    # Going back to the dataset, here we create a subclass that splits it into
    # standard training, validation, testing subsets.
    #


    # Create training and testing split of the data. We do not use validation in this tutorial.
    train_set = data.SubsetSC("training")
    test_set = data.SubsetSC("testing")

    waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]


    ######################################################################
    # A data point in the SPEECHCOMMANDS dataset is a tuple made of a waveform
    # (the audio signal), the sample rate, the utterance (label), the ID of
    # the speaker, the number of the utterance.
    #

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    # plt.plot(waveform.t().numpy());


    ######################################################################
    # Let’s find the list of labels available in the dataset.
    #

    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))
    labels


    ######################################################################
    # The 35 audio labels are commands that are said by users. The first few
    # files are people saying “marvin”.
    #

    waveform_first, *_ = train_set[0]
    # ipd.Audio(waveform_first.numpy(), rate=sample_rate)

    waveform_second, *_ = train_set[1]
    # ipd.Audio(waveform_second.numpy(), rate=sample_rate)


    ######################################################################
    # The last file is someone saying “visual”.
    #

    waveform_last, *_ = train_set[-1]
    # ipd.Audio(waveform_last.numpy(), rate=sample_rate)


    ######################################################################
    # Formatting the Data
    # -------------------
    #
    # This is a good place to apply transformations to the data. For the
    # waveform, we downsample the audio for faster processing without losing
    # too much of the classification power.
    #
    # We don’t need to apply other transformations here. It is common for some
    # datasets though to have to reduce the number of channels (say from
    # stereo to mono) by either taking the mean along the channel dimension,
    # or simply keeping only one of the channels. Since SpeechCommands uses a
    # single channel for audio, this is not needed here.
    #

    transform = data.get_transform(sample_rate)
    transformed = transform(waveform)

    # ipd.Audio(transformed.numpy(), rate=new_sample_rate)


    ######################################################################
    # We are encoding each word using its index in the list of labels.
    #

    word_start = "yes"
    index = data.label_to_index(word_start, labels)
    word_recovered = data.index_to_label(index, labels)

    print(word_start, "-->", index, "-->", word_recovered)


    ######################################################################
    # To turn a list of data point made of audio recordings and utterances
    # into two batched tensors for the model, we implement a collate function
    # which is used by the PyTorch DataLoader that allows us to iterate over a
    # dataset by batches. Please see `the
    # documentation <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`__
    # for more information about working with a collate function.
    #
    # In the collate function, we also apply the resampling, and the text
    # encoding.
    #


    def pad_sequence(batch):
        # Make all tensor in a batch the same length by padding with zeros
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)


    def collate_fn(batch):

        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors, targets = [], []

        # Gather in lists, and encode labels as indices
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [data.label_to_index(label, labels)]

        # Group the list of tensors into a batched tensor
        tensors = pad_sequence(tensors)
        targets = torch.stack(targets)

        return tensors, targets


    batch_size = 256

    if device == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # set up model
    audio_model = model.M5(n_input=transformed.shape[0], n_output=len(labels))
    audio_model.to(device)
    print(model)


    def count_parameters(audio_model):
        return sum(p.numel() for p in audio_model.parameters() if p.requires_grad)


    n = count_parameters(audio_model)
    print("Number of parameters: %s" % n)


    ######################################################################
    # We will use the same optimization technique used in the paper, an Adam
    # optimizer with weight decay set to 0.0001. At first, we will train with
    # a learning rate of 0.01, but we will use a ``scheduler`` to decrease it
    # to 0.001 during training after 20 epochs.
    #

    optimizer = optim.Adam(audio_model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10


    ######################################################################
    # Training and Testing the Network
    # --------------------------------
    #
    # Now let’s define a training function that will feed our training data
    # into the model and perform the backward pass and optimization steps. For
    # training, the loss we will use is the negative log-likelihood. The
    # network will then be tested after each epoch to see how the accuracy
    # varies during the training.
    #


    def train(audio_model, epoch, log_interval):
        audio_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = transform(data)
            output = audio_model(data)

            # negative log-likelihood for a tensor of size (batch x 1 x n_output)
            loss = F.nll_loss(output.squeeze(), target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print training stats
            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

            # update progress bar
            pbar.update(pbar_update)
            # record loss
            losses.append(loss.item())

        tensorboard_writer.add_scalar("train loss", losses[-1], epoch)

        # save model
        checkpoints_path = os.path.join(save_path, "checkpoints")
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        model_save_path = os.path.join(checkpoints_path, "model_{}.pt".format(epoch))
        print("saving to", model_save_path, "...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': audio_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_save_path)
        print("...saved")


    ######################################################################
    # Now that we have a training function, we need to make one for testing
    # the networks accuracy. We will set the model to ``eval()`` mode and then
    # run inference on the test dataset. Calling ``eval()`` sets the training
    # variable in all modules in the network to false. Certain layers like
    # batch normalization and dropout layers behave differently during
    # training so this step is crucial for getting correct results.
    #


    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()


    def test(audio_model, epoch):
        audio_model.eval()
        correct = 0
        for data, target in test_loader:

            data = data.to(device)
            target = target.to(device)

            # apply transform and model on whole batch directly on device
            data = transform(data)
            output = audio_model(data)

            pred = model.get_likely_index(output)
            correct += number_of_correct(pred, target)

            # update progress bar
            pbar.update(pbar_update)

        print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
        tensorboard_writer.add_scalar("test accuracy", correct / len(test_loader.dataset), epoch)


    # set up tensorboard

    tensorboard_writer = SummaryWriter(save_path)
    # tensorboard_writer.add_graph(model, transform(train_set[0][0]))


    ######################################################################
    # Finally, we can train and test the network. We will train the network
    # for ten epochs then reduce the learn rate and train for ten more epochs.
    # The network will be tested after each epoch to see how the accuracy
    # varies during the training.
    #

    log_interval = 20
    n_epoch = 50

    pbar_update = 1 / (len(train_loader) + len(test_loader))
    losses = []

    # The transform needs to live on the same device as the model and the data.
    transform = transform.to(device)
    with tqdm(total=n_epoch) as pbar:
        for epoch in range(1, n_epoch + 1):
            train(audio_model, epoch, log_interval)
            test(audio_model, epoch)
            scheduler.step()

    tensorboard_writer.close()

    # Let's plot the training loss versus the number of iteration.
    # plt.plot(losses);
    # plt.title("training loss");
