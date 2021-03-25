import aquariumlearning as al
import json
import os
import pickle
import random
import string
import time
from tqdm import tqdm

import data


AL_PROJECT = "speech_commands"
AL_DATASET = 'dataset_v1'
AL_INFERENCES = "test_1616737691_model_50_v1"

train_set = data.SubsetSC("training")
test_set = data.SubsetSC("testing")
classnames = sorted(list(set(datapoint[2] for datapoint in train_set)))

al_client = al.Client()
al_client.set_credentials(api_key=os.getenv('AQUARIUM_KEY'))

inferences = None
with open("inferences.pickle", "rb") as f:
    inferences = pickle.load(f)

embeddings = None
with open("embeddings.pickle", "rb") as f:
    embeddings = pickle.load(f)

assert len(inferences) == len(embeddings)

inferences_map = {}
embeddings_map = {}
for i in range(len(inferences)):
    (frame_id, pred_class, confidence, dataset_split) = inferences[i]
    inferences_map[frame_id] = inferences[i]
    embeddings_map[frame_id] = embeddings[i]

prev_frame_ids = set()

def process_frame(al_dataset, al_inferences, i, row, dataset_split):
    (waveform, sample_rate, utterance, speaker_id, utterance_number) = row

    frame_id = "{}_{}_{}".format(speaker_id, utterance_number, utterance)
    if frame_id not in prev_frame_ids:
        prev_frame_ids.add(frame_id)
    else:
        print ("DUPLICATE FRAME ID!", frame_id, dataset_split)
        raise Exception()
        return

    ###
    # add labeled datapoint
    ###

    al_labeled_frame = al.LabeledFrame(frame_id=frame_id)

    # Add arbitrary metadata, such as the train vs test split
    al_labeled_frame.add_user_metadata('speaker_id', speaker_id)
    al_labeled_frame.add_user_metadata('utterance_number', utterance_number)
    al_labeled_frame.add_user_metadata('waveform_shape', waveform.shape[1])
    al_labeled_frame.add_user_metadata('split', dataset_split)

    # Add an image to the frame
    data_url = "https://storage.googleapis.com/aquarium-public/datasets/speechCommands/"
    image_url = data_url + "spectrograms/{}/{}_nohash_{}.png".format(utterance, speaker_id, utterance_number)
    al_labeled_frame.add_image(sensor_id='spectrogram', image_url=image_url)

    audio_url = data_url + "speech_commands_v0.02/{}/{}_nohash_{}.wav".format(utterance, speaker_id, utterance_number)
    al_labeled_frame.add_audio(sensor_id='waveform', audio_url=audio_url)

    # Add the ground truth classification label to the frame
    label_id = frame_id + '_gt'
    al_labeled_frame.add_label_2d_classification(
        sensor_id='spectrogram',
        label_id=label_id,
        classification=utterance
    )

    al_labeled_frame.add_frame_embedding(embedding=embeddings_map[frame_id])
    al_dataset.add_frame(al_labeled_frame)

    ###
    # add inference
    ###
    (frame_id, pred_class, confidence, _) = inferences_map[frame_id]
    al_inf_frame = al.InferencesFrame(frame_id=frame_id)

    inf_label_id = frame_id + "_inf"
    al_inf_frame.add_inference_2d_classification(
        sensor_id="spectrogram",
        label_id=inf_label_id,
        classification=pred_class,
        confidence=float(confidence)
    )

    al_inf_frame.add_frame_embedding(embedding=embeddings_map[frame_id])
    al_inferences.add_frame(al_inf_frame)

al_dataset = al.LabeledDataset()
al_inferences = al.Inferences()
for i, row in tqdm(enumerate(train_set)):
    process_frame(al_dataset, al_inferences, i, row, "train")

for i, row in tqdm(enumerate(test_set)):
    process_frame(al_dataset, al_inferences, i, row, "test")

al_client.create_project(
    AL_PROJECT,
    al.LabelClassMap.from_classnames(classnames),
    primary_task="CLASSIFICATION"
)

al_client.create_dataset(
    AL_PROJECT,
    AL_DATASET,
    dataset=al_dataset,
    # Poll for completion of the processing job
    wait_until_finish=True,
    # Preview the first frame before submission to catch mistakes
    preview_first_frame=True
)

al_client.create_inferences(
    AL_PROJECT,
    AL_DATASET,
    inferences=al_inferences,
    inferences_id=AL_INFERENCES,
    # Poll for completion of the processing job
    wait_until_finish=True,
)

