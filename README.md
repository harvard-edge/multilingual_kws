
### Common Voice

Download the english dataset (38GB) from https://voice.mozilla.org/en/datasets

### DeepSpeech

Download the pretrained english model from https://github.com/mozilla/DeepSpeech

```bash
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.0/deepspeech-0.7.0-models.pbmm
```

### Running the demo for word-splitting

If you do not have `conda` installed for python dependency management, first install [`miniconda`](https://docs.conda.io/en/latest/miniconda.html). After cloning this repository, run the following commands:

**Note:** If you do not have a GPU, edit `environment.yml` and replace `deepspeech-gpu` with `deepspeech` first.

```bash
cd tinyspeech/
conda env create -f environment.yml
conda activate deepspeech
mkdir tmp # or edit tmp_wav_filepath
jupyter lab SplitDemo.ipynb
```
You will also need to edit the `clips_path` and `read_csv("/path/to/train.tsv")` arguments to point to your download location for the Common Voice dataset.

