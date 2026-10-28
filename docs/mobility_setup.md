## Configure SpectralGPT

```bash
wget https://zenodo.org/records/13139925/files/SpectralGPT.pth?download=1
mv SpectralGPT.pth?download=1 SpectralGPT.pth
```

Set `PATHS_SPECTRAL_PRETRAINED` in `.env` to the downloaded checkpoint.

## Install requirements

```bash
uv sync            # or: pip install -r requirements.txt && pip install -e .
```

`requirements.lock.txt` holds the exact environment of the cluster runs (Python 3.11.9).

## Copy the task-tuning heads

The merge experiments need the heads of a task-tuning run
(`<country>_lora.safetensors` and `<country>_fc.safetensors`):

```powershell
pscp -r user@erde.rsim.tu-berlin.de:/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/saved_models/epoch20/task_tuning .\
```

Set `PATHS_TASK_HEADS` in `.env` (or `merge.task_heads_dir` in the config) to this folder.

## Use saved datasets instead of the full BigEarthNet V2

- On the machine with the LMDB, run `python scripts/save_datasets.py --config configs/default.yaml --out ~/saved_datasets`
- Transfer `~/saved_datasets` to the new machine
- In the experiment config set:

```yaml
data:
  saved_datasets_dir: "~/saved_datasets"
```

Saved images are not augmented: the random training augmentations are lost when data is saved.
