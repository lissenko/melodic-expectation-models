# Evaluating Gestalt-Like, Multiple Viewpoint, and Neural Models of Melodic Expectation

Code and data of the paper by Lissenko, Rocamora, and Anglada-Tort. The repository can be used to:

- predict melodies with the LSTM and Transformer networks of the paper;
- train new networks on a corpus of MIDI files;
- evaluate any model of melodic expectation on four behavioral datasets.

## Installation

```bash
pip install -r requirements.txt
```

## Pretrained networks

The 30 LSTMs and 30 Transformers of the paper are attached to the [latest release](../../releases/latest):

```bash
gh release download --pattern "lstm_*" --dir checkpoints/lstm/
gh release download --pattern "transformer_*" --dir checkpoints/transformer/
```

## Predicting a melody

```bash
python inference.py my_melody.mid transformer checkpoints/transformer/
```

```python
from inference import load_ensemble, predict_ensemble

ensemble = load_ensemble("transformer", "checkpoints/transformer/")
result = predict_ensemble(ensemble, "my_melody.mid")
result["ics"]            # information content of each note (bits)
result["probabilities"]  # distribution over the 128 pitches before each note
```

The key, needed for the interval from the tonic, is estimated with music21.

## Training a network

```bash
python training/train_transformer.py --data my_midi_folder/ --out checkpoints/my_transformer/transformer_01.pth --seed 1
python training/train_lstm.py --data my_midi_folder/ --out checkpoints/my_lstm/lstm_01.pth --seed 1
```

The defaults are those of the paper: architecture, learning rate, and number of epochs. Melodies are
grouped into families of near-duplicates and 400 + 1,000 melodies are held out by family
(`training/split.py`); `--no_holdout` trains on every melody. `--features` selects the input features:

| Feature | IDyOM viewpoint | |
|---|---|---|
| `pitch` | `cpitch` | MIDI pitch |
| `pitch_class` | `cpitch-class` | pitch class |
| `tessitura` | `tessitura` | low, middle, or high register |
| `interval` | `cpint` | interval from the previous note |
| `contour` | `contour` | down, same, or up |
| `cpintfip` | `cpintfip` | interval from the first note |
| `cpintfref` | `cpintfref` | interval from the tonic |
| `dur` | `dur` | duration |
| `bioi` | `bioi` | inter-onset interval |
| `bioi_ratio` | `bioi-ratio-q` | ratio to the previous inter-onset interval, rounded |
| `bioi_contour` | `bioi-contour-q` | shorter, same, or longer inter-onset interval |

Features are computed as IDyOM computes the corresponding viewpoints. The two rounded rhythm viewpoints
are defined for IDyOM in `idyom/viewpoints.lisp`. Durations and inter-onset intervals are in IDyOM's basic
time units at a timebase of 500, with MIDI files read at their nominal tempo.

## Evaluating a model

`evaluate.py` gives the variance in participants' responses explained by each model, and the variance unique
to each of two models and common to both:

```bash
python evaluate.py results/predictions/idyom results/predictions/transformer my_predictions/
```

| Dataset | Task | Stimuli |
|---|---|---|
| Cuddy and Lunney (1995) | continuation ratings | 8 two-tone contexts, 25 continuations each |
| Schellenberg (1996) | continuation ratings | 8 folk-song fragments, 15 continuations each |
| Manzara et al. (1992) | bets on successive notes | 2 chorale melodies, 86 notes |
| Fogel et al. (2015) | sung continuations | 90 melodies, 7 scale degrees each |

A model's predictions are one folder with one file per dataset, `<dataset>.json`, keyed by the name of the
stimulus in `data/stimuli/<dataset>/` (for Schellenberg, the number of the fragment):

```json
{"AC01": {"last_note_probs": [128 probabilities of the next pitch after the context]}}
```

Manzara et al. use `{"short": {"ics": [...]}, "long": {"ics": [...]}}`, the information content in bits of
every note of Chorales 151 and 61. For a network, `evaluation/predict_stimuli.py` writes this folder:

```bash
python evaluation/predict_stimuli.py --arch transformer --checkpoints checkpoints/transformer/ --out my_predictions/
```

`results/predictions/` holds the predictions of the four models of the paper (Temperley's model, IDyOM,
LSTM, Transformer). The stimuli in `data/stimuli/` are timed as participants heard them, and
`data/stimulus_keys.json` gives their notated keys. The values of Manzara et al. were read from
Pearce (2005, Figures 8.7 and 8.8).

## License

MIT.
