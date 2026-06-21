# Deep Learning, Statistical Learning, and Rule-Based Models of Human Melodic Expectation

## Installation

```bash
pip install -r requirements.txt
```
---

## Pretrained model weights

The 60 model checkpoints are attached to the
[latest GitHub release](../../releases/latest): 30 LSTM (`lstm_*.pth`) and
30 Transformer (`transformer_*.pth`) files.

Download them into place with the GitHub CLI (run from the repo root):

```bash
gh release download --pattern "lstm_*.pth"  --dir checkpoints/lstm/
gh release download --pattern "transformer_*.pth" --dir checkpoints/transformer/
```

Or download the files from the release page and put `lstm_*.pth` in
`checkpoints/lstm/` and `transformer_*.pth` in `checkpoints/transformer/`.

---

## Quick start: running inference on a MIDI file

### Ensemble checkpoint

```python
from inference import load_ensemble, predict_ensemble

# Load all 30 checkpoints
ensemble = load_ensemble("lstm", "checkpoints/lstm/")
result = predict_ensemble(ensemble, "my_melody.mid")

print(result["ics"])           # information content per note
print(result["entropies"])     # entropy per note
print(result["mdwics"])        # melody-duration-weighted IC
print(result["probabilities"]) # full 128-pitch distribution per note
```

The same interface works for the Transformer:

```python
ensemble = load_ensemble("transformer", "checkpoints/transformer/")
result = predict_ensemble(ensemble, "my_melody.mid")
```

### Single checkpoint

```python
from inference import load_single, predict_single

model, norm = load_single("lstm", "checkpoints/lstm/lstm_01.pth")
result = predict_single(model, norm, "my_melody.mid")
```

### Command-line

```bash
# Ensemble
python inference.py my_melody.mid lstm checkpoints/lstm/

# Single checkpoint
python inference.py my_melody.mid transformer checkpoints/transformer/transformer_01.pth
```

---

## Temperley model

```python
from models.temperley import estimate_parameters, predict

params = estimate_parameters("/path/to/training/midi/")
result = predict("my_melody.mid", params)
print(result["ics"])
```

---

## Reproducing paper results

Precomputed model predictions are in `data/predictions/{lstm,transformer,temperley,idyom}/`.  
Human data are in `data/human/`. Raw IDyOM `.dat` outputs are in
`data/predictions/idyom/raw/` (run `python data/predictions/preprocess_idyom.py` to
rebuild the IDyOM JSONs from them).

```bash
# Individual experiments
python evaluate.py --experiment manzara 
python evaluate.py --experiment cuddy_lunney
python evaluate.py --experiment schellenberg
python evaluate.py --experiment fogel 
python evaluate.py --experiment model_correlation

# All at once
python evaluate.py
```

Results are printed to the console and also written to `evaluation_results.json`
(override with `-o`)

---

## Training from scratch

The paper ensembles 30 models per architecture, each trained with a different `--seed`.

### LSTM

```bash
python training/train_lstm.py \
    --data /path/to/training/midi/ \
    --out  checkpoints/lstm/lstm_01.pth \
    --seed 42
```

Default hyperparameters match the paper:
`hidden_size=720, num_layers=2, dropout=0.5, epochs=9, lr=1e-3`

Selected features: `pitch_class, contour, cpintfip, scale_degree, key_membership, beat_position, duration`

### Transformer

```bash
python training/train_transformer.py \
    --data /path/to/training/midi/ \
    --out  checkpoints/transformer/transformer_01.pth \
    --seed 1
```

Default hyperparameters match the paper:
`d_model=320, nhead=8, num_layers=5, dim_feedforward=1280, dropout=0.1, epochs=9, lr=1e-4`

Selected features: `pitch, contour, duration, scale_degree, ioi`

---

## Feature descriptions

| Feature | Dim | Description |
|---|---|---|
| `pitch` | 128 | One-hot MIDI pitch |
| `pitch_class` | 12 | One-hot pitch class |
| `interval` | 73 | One-hot interval from previous pitch (-36 to +36) |
| `contour` | 3 | Melodic direction: up / same / down |
| `cpintfip` | 73 | Pitch interval from the first note of the melody |
| `scale_degree` | 7 | One-hot diatonic scale degree in the inferred key |
| `key_membership` | 1 | Binary: note is in the estimated key |
| `register` | 3 | Low / mid / high pitch register |
| `onset` | 1 | Normalised note onset time |
| `duration` | 1 | Normalised note duration |
| `symbolic_duration` | 21 | One-hot nearest nominal note type |
| `ioi` | 1 | Normalised inter-onset interval |
| `beat_position` | 1 | Fractional position within the beat |
| `phrase` | 2 | One-hot phrase-boundary marker |
| `is_repeated_pitch` | 1 | Binary: same pitch as the previous note |

<!-- --- -->

<!-- ## Citation -->

<!-- ```bibtex -->
<!-- @article{lissenko2025melodicexpectation, -->
<!--   title   = {Deep Learning, Statistical Learning, and Rule-Based Models of Human Melodic Expectation}, -->
<!--   author  = {Lissenko, Tanguy and Rocamora, Mart{\'i}n and Anglada-Tort, Manuel}, -->
<!--   year    = {2025} -->
<!-- } -->
<!-- ``` -->


