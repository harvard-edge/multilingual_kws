import os
import logging
from typing import Dict, List, Optional

import glob
import numpy as np
import tensorflow as tf

import sys

import multilingual_kws.embedding.input_data as input_data


def transfer_learn(
    target,
    train_files,
    val_files,
    unknown_files,
    num_epochs,
    num_batches,
    batch_size,
    primary_lr,
    backprop_into_embedding,
    embedding_lr,
    model_settings: Dict,
    base_model_path: os.PathLike,
    base_model_output: str,
    UNKNOWN_PERCENTAGE: float = 50.0,
    bg_datadir: os.PathLike = "/home/mark/tinyspeech_harvard/speech_commands/_background_noise_/",
    csvlog_dest: Optional[os.PathLike] = None,
    verbose=1,
):
    """this only words for single-target models: see audio_dataset and CATEGORIES"""

    tf.get_logger().setLevel(logging.ERROR)
    base_model = tf.keras.models.load_model(base_model_path)
    tf.get_logger().setLevel(logging.INFO)
    xfer = tf.keras.models.Model(
        name="TransferLearnedModel",
        inputs=base_model.inputs,
        outputs=base_model.get_layer(name=base_model_output).output,
    )
    xfer.trainable = False

    # dont use softmax unless losses from_logits=False
    CATEGORIES = 3  # silence + unknown + target_keyword
    xfer = tf.keras.models.Sequential(
        [
            xfer,
            tf.keras.layers.Dense(units=18, activation="tanh"),
            tf.keras.layers.Dense(units=CATEGORIES, activation="softmax"),
        ]
    )

    xfer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=primary_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # TODO(mmaz): use keras class weights?
    audio_dataset = input_data.AudioDataset(
        model_settings=model_settings,
        commands=[target],
        background_data_dir=bg_datadir,
        unknown_files=unknown_files,
        unknown_percentage=UNKNOWN_PERCENTAGE,
        spec_aug_params=input_data.SpecAugParams(percentage=80),
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    init_train_ds = audio_dataset.init_single_target(
        AUTOTUNE, train_files, is_training=True
    )
    init_val_ds = audio_dataset.init_single_target(
        AUTOTUNE, val_files, is_training=False
    )
    train_ds = init_train_ds.shuffle(buffer_size=1000).repeat().batch(batch_size)
    val_ds = init_val_ds.batch(batch_size)

    if csvlog_dest is not None:
        callbacks = [tf.keras.callbacks.CSVLogger(csvlog_dest, append=False)]
    else:
        callbacks = []

    history = xfer.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=batch_size * num_batches,
        epochs=num_epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    if backprop_into_embedding:
        # https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/#transfer-learning-from-pretrained-weights
        # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
        for layer in xfer.layers[-20:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        xfer.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=embedding_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        history = xfer.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=batch_size * num_batches,
            epochs=num_epochs,
            callbacks=callbacks,
        )

    va = history.history["val_accuracy"][-1]
    name = f"xfer_epochs_{num_epochs}_bs_{batch_size}_nbs_{num_batches}_val_acc_{va:0.2f}_target_{target}"
    details = dict(
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_batches=num_batches,
        val_accuracy=va,
        target=target,
    )
    return name, xfer, details


def random_sample_transfer_models(
    NUM_MODELS,
    N_SHOTS,
    VAL_UTTERANCES,
    oov_words,
    dest_dir,
    unknown_files,
    EPOCHS,
    data_dir,
    model_settings,
    base_model_path: os.PathLike,
    base_model_output="dense_2",
    UNKNOWN_PERCENTAGE=50.0,
    NUM_BATCHES=1,
    bg_datadir="/home/mark/tinyspeech_harvard/frequent_words/en/clips/_background_noise_/",
):
    assert os.path.isdir(dest_dir), f"dest dir {dest_dir} not found"
    models = np.random.choice(oov_words, NUM_MODELS, replace=False)

    for target in models:
        wavs = glob.glob(data_dir + target + "/*.wav")
        selected = np.random.choice(wavs, N_SHOTS + VAL_UTTERANCES, replace=False)

        train_files = selected[:N_SHOTS]
        np.random.shuffle(train_files)
        val_files = selected[N_SHOTS:]

        print(len(train_files), "shot:", target)

        utterances_fn = target + "_utterances.txt"
        utterances = dest_dir + utterances_fn
        print("saving", utterances)
        with open(utterances, "w") as fh:
            fh.write("\n".join(train_files))

        transfer_learn(
            dest_dir=dest_dir,
            target=target,
            train_files=train_files,
            val_files=val_files,
            unknown_files=unknown_files,
            EPOCHS=EPOCHS,
            model_settings=model_settings,
            base_model_path=base_model_path,
            base_model_output=base_model_output,
            UNKNOWN_PERCENTAGE=UNKNOWN_PERCENTAGE,
            NUM_BATCHES=NUM_BATCHES,
            bg_datadir=bg_datadir,
        )


def evaluate_fast_multiclass(
    words_to_evaluate: List[str],
    target_id: int,
    data_dir: os.PathLike,
    utterances_per_word: int,
    model: tf.keras.Model,
    model_settings: Dict,
):
    correct_confidences = []
    incorrect_confidences = []

    specs = []
    for word in words_to_evaluate:
        wavs = glob.glob(data_dir + word + "/*.wav")
        if len(wavs) > utterances_per_word:
            fs = np.random.choice(wavs, utterances_per_word, replace=False)
        else:
            print("using all wavs for ", word)
            fs = wavs
        specs.extend([input_data.file2spec(model_settings, f) for f in fs])
    specs = np.array(specs)
    preds = model.predict(np.expand_dims(specs, -1))

    # softmaxes = np.max(preds,axis=1)
    # unknown_other_words_confidences.extend(softmaxes.tolist())
    cols = np.argmax(preds, axis=1)
    # figure out how to fancy-index this later
    for row, col in enumerate(cols):
        confidence = preds[row][col]
        if col == target_id:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
    return {
        "correct": correct_confidences,
        "incorrect": incorrect_confidences,
    }


def evaluate_fast_single_target(
    words_to_evaluate: List[str],
    target_id: int,
    data_dir: os.PathLike,
    utterances_per_word: int,
    model: tf.keras.Model,
    model_settings: Dict,
):
    specs = []
    for word in words_to_evaluate:
        wavs = glob.glob(data_dir + word + "/*.wav")
        if len(wavs) > utterances_per_word:
            fs = np.random.choice(wavs, utterances_per_word, replace=False)
        else:
            print("using all wavs for ", word)
            fs = wavs
        specs.extend([input_data.file2spec(model_settings, f) for f in fs])
    specs = np.array(specs)
    preds = model.predict(np.expand_dims(specs, -1))
    return preds[:, target_id], preds


def evaluate_files_multiclass(
    files_to_evaluate: List[os.PathLike],
    target_id: int,
    model: tf.keras.Model,
    model_settings: Dict,
):
    correct_confidences = []
    incorrect_confidences = []

    specs = [input_data.file2spec(model_settings, f) for f in files_to_evaluate]
    specs = np.array(specs)
    preds = model.predict(np.expand_dims(specs, -1))

    # softmaxes = np.max(preds,axis=1)
    # unknown_other_words_confidences.extend(softmaxes.tolist())
    cols = np.argmax(preds, axis=1)
    # figure out how to fancy-index this later
    for row, col in enumerate(cols):
        confidence = preds[row][col]
        if col == target_id:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
    return dict(correct=correct_confidences, incorrect=incorrect_confidences)


def evaluate_files_single_target(
    files_to_evaluate: List[os.PathLike],
    target_id: int,
    model: tf.keras.Model,
    model_settings: Dict,
):
    specs = [input_data.file2spec(model_settings, f) for f in files_to_evaluate]
    specs = np.array(specs)
    preds = model.predict(np.expand_dims(specs, -1))
    return preds[:, target_id], preds


def evaluate_and_track(
    words_to_evaluate: List[str],
    target_id: int,
    data_dir: os.PathLike,
    utterances_per_word: int,
    model: tf.keras.Model,
    model_settings: Dict,
):
    # TODO(mmaz) rewrite and combine with evaluate_fast
    raise ValueError("this only works for multiclass, see other evaluation functions")

    correct_confidences = []
    incorrect_confidences = []
    track_correct = {}
    track_incorrect = {}

    for word in words_to_evaluate:
        fs = np.random.choice(
            glob.glob(data_dir + word + "/*.wav"), utterances_per_word, replace=False
        )

        track_correct[word] = []
        track_incorrect[word] = []

        specs = np.array([input_data.file2spec(model_settings, f) for f in fs])
        preds = model.predict(np.expand_dims(specs, -1))

        # softmaxes = np.max(preds,axis=1)
        # unknown_other_words_confidences.extend(softmaxes.tolist())
        cols = np.argmax(preds, axis=1)
        # figure out how to fancy-index this later
        for row, col in enumerate(cols):
            confidence = preds[row][col]
            if col == target_id:
                correct_confidences.append(confidence)
                track_correct[word].append(confidence)
            else:
                incorrect_confidences.append(confidence)
                track_incorrect[word].append(confidence)
    return {
        "correct": correct_confidences,
        "incorrect": incorrect_confidences,
        "track_correct": track_correct,
        "track_incorrect": track_incorrect,
    }
