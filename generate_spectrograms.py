import librosa
import matplotlib
import matplotlib.pyplot as plt
import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

import data

spectrogram_path = "SpeechCommands/spectrograms"

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None, save_path=None):
    fig, axs = plt.subplots(1, 1)
    # axs.set_title(title or 'Spectrogram (db)')
    # axs.set_ylabel(ylabel)
    # axs.set_xlabel('frame')
    plt.axis("off")
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
      axs.set_xlim((0, xmax))
    # fig.colorbar(im, ax=axs)
    # plt.show(block=False)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", transparent=True, pad_inches=0)
    else:
        plt.show()

    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = data.SubsetSC()

    waveform, sample_rate, utterance, *_ = train_set[-1]
    transform = data.get_transform(sample_rate)
    transform = transform.to(device)
    spectrograms = []
    print("generating spectrograms")
    for i, (waveform, sample_rate, utterance, speaker_id, utterance_number) in tqdm(enumerate(train_set)):
        # print(i, sample_rate, utterance, speaker_id, utterance_number)
        waveform = waveform.to(device)
        waveform = transform(waveform)

        n_fft = 1024
        win_length = None
        hop_length = 512
        n_mels = 128

        mel_spectrogram = T.MelSpectrogram(
            sample_rate=8000,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm='slaney',
            onesided=True,
            n_mels=n_mels,
        )
        mel_spectrogram = mel_spectrogram.to(device)

        melspec = mel_spectrogram(waveform)
        spectrograms.append(melspec)

    print("bringing back to cpu")
    spectrograms = [spec.cpu() for spec in spectrograms]

    def generate_figure(row):
        i, (waveform, sample_rate, utterance, speaker_id, utterance_number) = row
        melspec = spectrograms[i]
        spectrogram_class_path = os.path.join(spectrogram_path, utterance)
        if not os.path.exists(spectrogram_class_path):
            os.makedirs(spectrogram_class_path)
        out_path = os.path.join(spectrogram_class_path + "/{}_nohash_{}.png".format(speaker_id, utterance_number))

        plot_spectrogram(
            melspec[0],
            title="MelSpectrogram - torchaudio",
            ylabel='mel freq',
            save_path=out_path
        )

    print("generating figures")
    # for i, (waveform, sample_rate, utterance, speaker_id, utterance_number) in tqdm(enumerate(train_set)):
    #     melspec = spectrograms[i].cpu()
    #     spectrogram_class_path = os.path.join(spectrogram_path, utterance)
    #     if not os.path.exists(spectrogram_class_path):
    #         os.makedirs(spectrogram_class_path)
    #     out_path = os.path.join(spectrogram_class_path + "/{}_nohash_{}.png".format(speaker_id, utterance_number))

    #     plot_spectrogram(
    #         melspec[0],
    #         title="MelSpectrogram - torchaudio",
    #         ylabel='mel freq',
    #         save_path=out_path
    #     )

    # for i, row in enumerate(train_set):
    #     generate_figure(i, row)

    from torch.multiprocessing import Pool
    torch.multiprocessing.set_sharing_strategy('file_system')

    with Pool() as pool:
        r = list(tqdm(pool.imap(generate_figure, enumerate(train_set))))
