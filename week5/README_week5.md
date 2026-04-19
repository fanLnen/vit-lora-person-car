# Week 5 - Initial Training and Hyperparameter Tuning

This week focuses on three tasks:

1. Small-subset quick test  
   Use 1%-5% of the training set and run 1-2 epochs to verify:
   - data loading
   - forward propagation
   - backward propagation
   - optimizer update
   - loss behavior

2. Learning rate finder  
   Increase the learning rate exponentially from `1e-7` to `10` over `100` iterations,
   then plot the learning rate vs. loss curve to determine a recommended LR range.

3. Hyperparameter sweep  
   Compare the following parameter combinations with the same LoRA architecture:
   - `(r=8, alpha=16, lr=1e-4)`
   - `(r=16, alpha=32, lr=3e-4)`
   - `(r=32, alpha=64, lr=1e-3)`
   - `(r=64, alpha=128, lr=3e-3)`

Target modules are fixed to:
`["query", "key", "value", "output.dense"]`

Optimizer:
`AdamW`

## Files

- `dataset.py`  
  Existing dataset loader from Week 4

- `utils.py`  
  Updated utility functions with parameterized `get_lora_config(...)`

- `week5_experiments.py`  
  Main script for quick test, LR finder, and hyperparameter sweep

## Output Files

All results are saved under:

`outputs_week5/`

Expected outputs:
- `quick_test_results.csv`
- `quick_test_summary.json`
- `lr_finder_results.csv`
- `lr_finder_curve.png`
- `lr_finder_summary.json`
- `hyperparameter_results.csv`
- `hyperparameter_epoch_results.csv`
- `best_hyperparameters.json`

## Example Commands

Run all experiments:
```bash
python week5_experiments.py --mode multilabel --run_all
```

Run only the quick test:
```bash
python week5_experiments.py --mode multilabel --run_quick_test
```

Run only the LR finder:
```bash
python week5_experiments.py --mode multilabel --run_lr_finder
```

Run only the hyperparameter sweep:
```bash
python week5_experiments.py --mode multilabel --run_sweep
```

For single-label mode:
```bash
python week5_experiments.py --mode singlelabel --run_all
```

## Notes

- The quick test and LR finder use a small subset for speed.
- The hyperparameter sweep also uses a small training subset by default (`5%`) for fast screening.
- Validation is always performed on the fixed validation set.
- The best parameter combination is selected by:
  1. highest validation accuracy
  2. lowest validation loss (tie-break)
- The selected best config can be used as the base setting for later full training.
