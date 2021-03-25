import pickle
import torch
from tqdm import tqdm

import data
import model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_set = data.SubsetSC("training")
    test_set = data.SubsetSC("testing")
    waveform, sample_rate, utterance, *_ = train_set[0]

    transform = data.get_transform(sample_rate)
    transform.to(device)

    transformed = transform(waveform)
    labels = sorted(list(set(datapoint[2] for datapoint in train_set)))

    audio_model = model.M5(n_input=transformed.shape[0], n_output=len(labels))
    checkpoint = torch.load("runs/test_1616737691/checkpoints/model_50.pt")
    audio_model.load_state_dict(checkpoint["model_state_dict"])
    audio_model.eval()
    audio_model.to(device)

    # embedding extraction
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    audio_model.pool4.register_forward_hook(get_activation("pool4"))

    def predict(tensor):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(device)
        tensor = transform(tensor)
        tensor = audio_model(tensor.unsqueeze(0))
        likely_ind = model.get_likely_index(tensor)
        out_label = data.index_to_label(likely_ind.squeeze(), labels)
        probs = torch.exp(tensor).squeeze()
        confidence = probs[likely_ind]
        embedding = activation["pool4"].flatten()

        return out_label, confidence.item(), embedding.cpu().tolist()

    out_rows = []
    out_embeddings = []

    def handle_row(i, row, dataset_split):
        (waveform, sample_rate, utterance, speaker_id, utterance_number) = row

        # each clip should be one second long, so each array should be
        # "sample_rate" elements long
        if waveform.shape[1] != sample_rate:
            # print("padding waveform")
            # print(waveform.shape, sample_rate, utterance)
            pad_amount = int((sample_rate - waveform.shape[1]) / 2)
            waveform = torch.nn.functional.pad(waveform, (pad_amount, pad_amount))

        frame_id = "{}_{}_{}".format(speaker_id, utterance_number, utterance)

        pred_class, confidence, embedding = predict(waveform)

        out_rows.append((frame_id, pred_class, str(confidence), dataset_split))
        out_embeddings.append((embedding))

    for i, row in tqdm(enumerate(train_set)):
        handle_row(i, row, "train")

    for i, row in tqdm(enumerate(test_set)):
        handle_row(i, row, "test")

    assert len(out_rows) == len(out_embeddings)

    with open("inferences.pickle", "wb") as f:
        pickle.dump(out_rows, f)

    with open("embeddings.pickle", "wb") as f:
        pickle.dump(out_embeddings, f)
