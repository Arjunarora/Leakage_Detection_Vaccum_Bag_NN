# Leakage Detection in Vacuum Bagging with Neural Networks

Locating a leak in a vacuum bag (used in composite manufacturing) from the flow rates of **four mass-flow controllers (MFC)** placed around the part. A neural network maps the four normalised flow rates to the 2-D leak coordinates `(y1, y2)`.

The problem has a built-in symmetry: rotating or mirroring the setup permutes the sensors and rotates the leak position the same way. The project compares a **standard MLP** with an **equivariant network** whose layers bake that symmetry in, and with **data augmentation** that adds the seven rotated/flipped copies of every example.

## Approach

1. **Sensor correction:** sensor 4 (MFC4) malfunctioned and reads systematically high. It is corrected by subtracting the mean of the other three sensors.
2. **Standard MLP** (Keras): baseline, then hyperparameter search with KerasTuner over width (32–512), depth (1–5) and learning rate. Loss: mean absolute error.
3. **Data augmentation:** each `(x, y)` is extended with rotations (90°, 180°, 270°) and vertical flips: `y90 = (y2, −y1)`, `y_flip = (−y1, y2)`.
4. **Equivariant network:** custom `EquivariantHiddenLayer` / `EquivariantOutputLayer` with tied weights, so predictions transform consistently under the symmetry group.
5. **Evaluation:** validation MAE plus hand-built symmetric test sets that check whether predictions rotate along with the inputs.

## Results (from the original run)

| Model | Training data | Best validation MAE |
|---|---|---|
| Standard MLP, tuned | 1000 samples | 0.174 |
| Standard MLP, tuned | 1000 samples + augmentation | 0.298 |

Augmentation *hurt* the tuned MLP in this run. This is very likely caused by a bug in the original augmentation code (see known issues below). The equivariant model was trained, but its validation scores weren't recorded in the notebook.

## Known issues (to-do)

- `augment_data` applies `np.rot90` to the whole data matrix, which transposes it (`(4, 1000)` instead of `(1000, 4)`). A rotation should instead **cyclically shift the four sensor columns** and rotate `y`. The correct shift direction depends on the sensor layout, so it's left for a re-run.
- `accuracy` is logged as a metric, but it is meaningless for regression. MAE is the metric that matters.
- `data/leakage_dataset_validation_1000.csv` isn't included in the repo.

## Repository structure

```
├── leakage_detection.ipynb   # full workflow: EDA, MLP, tuning, augmentation, equivariant model
├── data/                     # training sets with 100 and 1000 samples
└── models/                   # saved models (created when you run the notebook, not tracked)
```

## Run

```bash
pip install -r requirements.txt
jupyter notebook leakage_detection.ipynb
```

Originally developed on Google Colab. The Drive-mount cell is commented out, and paths point to `data/` and `models/`.

## License

MIT
