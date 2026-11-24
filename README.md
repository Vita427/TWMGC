# TWMGC

Reference implementation of **Multi-view Graph Clustering Based on Triple-Contrast Learning and
Weighted Cross-View Feature Fusion (TWMGC)**.


## Repository layout

```
TWMGC/
├── main.py                     entry point: setup, data, diffusion, model, training
├── train.py                    training loop, evaluation, timing and memory report
├── DCRN.py                     AE / GAE encoders and decoders, attention, soft assignment
├── MSC.py                      MLFA: node-wise weighting of the four view-specific embeddings
├── GCN.py                      graph propagation layer used by the GAE
├── utils.py                    data loading, normalisation, diffusion, losses, metrics
├── logger.py                   console + file logger
├── opt.py                      command-line arguments
├── run_all.sh                  runs every dataset with ten seeds
└── DCRN_pretrain/
    ├── dataset/                data go here
    ├── pretrain_ae/            AE pretraining            → <dataset>_ae.pkl
    ├── pretrain_gae/           GAE pretraining           → <dataset>_gae.pkl
    └── pretrain/              joint DFCN-style pretraining → <dataset>_pretrain.pkl
```

## Environment

```
python >= 3.8
pip install -r requirements.txt
```


## Pretraining

Run from inside the corresponding directory (each script writes its `*.pkl` next to itself):

```
cd DCRN_pretrain/pretrain_ae  && python main.py --name acm
cd DCRN_pretrain/pretrain_gae && python main.py --name acm
cd DCRN_pretrain/pretrain     && python main.py --name acm --epoch 100
```

`--epoch 100` matches the 100 pretraining iterations of Section 3.2 of the paper.

## Training

```
python main.py --name acm --seed 0
```

The per-dataset settings of Section 3.2 of the paper are stored in `utils.setup()` and are used by
default; every weight and the learning rate can be overridden from the command line, for example

```
python main.py --name cocs --seed 0 --l1 1e-2 --l2 1e-2 --l3 1e-1 --lambda_value 1 --gamma_value 1e-3 --lr 7e-4
```

## Output

For every run the script prints ACC, NMI, ARI and F1 at the best epoch, writes the log to
`log/<dataset>==<timestamp>.log`, appends one row per run to `benchmark_results.csv`
(time, peak GPU memory, parameter count) and saves the final representation and the predicted
labels to `output/<dataset>_{fea,label}.xlsx`.

