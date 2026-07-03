Long read ahead)))- many personal thoughts and emotions.
All below is only my personal opinion and view, and when I say 'you should' or 'you must,' of course it's just my view.

Thanks Kaggle team and CMI for such great competition - the data is exceptional and was so much fun to work with it!

When you start working on competition, you should think from the start about several aspects:

The code organization
Versioning
Experiments tracking
Results repeatability
Code organization
Normally my ML experiments follow such a structure (that is enough and good for me).

data/: Contains all data files

raw/: Original, unmodified data files
processed/: Data after various processing steps
base/: Basic processed data (e.g., cleaned, filtered)
fe/: Feature-engineered data
temp/: Temporary data files that don't need to be version controlled
checkpoits/: To store weights and runs

notebooks/: Jupyter notebooks

public/: Notebooks that were publicly shared (of course you should check what other people did)
eda/: Data analysis (complex or single shot data checks)
old/: My TRASH folder where I drop copy of copy of copy notebooks
models/: Main folder where experiments live separated by models types
lgb/:
lstm/:
xxx/:
src/: Source code organized by functionality

data/: Code for data loading and processing
fe/: Feature engineering code
models/: Isolated code separated by model type
lgb/: LightGBM models
lstm/: LSTM model files
pca/: PCA model files
temp/: Temporary code files and test scripts
utils/: Utility functions and helper modules
history/: Historical project data and records / if no git in project doing src dumps here

tb/: Tensorboard tracking logs

tests/: Unit tests and integration tests

I really advise you to use some version control (git). It will greatly help you to navigate through your thoughts during the competition 'marathon' (normally 3 months).

The Plan
So you downloaded the data - so did I. Did some minimal data checks and started thinking about the plan.
The goal was GOLD - any other result would be a failure)))).

We have 2 types of Data, so I will need good models for each branch IMU and TOF. IMU is easy just 7 features - so will start with it.
It's a solo competition, so I need to Ensemble in the end with different architectures and model types.
Will need NNs for sure - any gradient boosting will be not enough.
Post-processing? Sure. Almost any competition needs PP. Also, F1 metric is good for hacking (some thresholds optimizations?).
Inference time limit. Data is sequence per sequence and small - time should not be an issue.
Models types - We have a temporal component (sequence) that should be used - lstm / gru?
We have 18 classes but also orientation / phase, and the main target that could split by components 'Forehead/zone' - 'pull hairline/action' - needs to be used for sure.
Just 9k sequences - Augmentation we need
First steps
I decided to start with lgbm (surprise)))) The previous CMI competition (that we won with @trasibulo) showed that gradient models could be competitive to NN in some tasks.
Did few experiments that show interesting details.

Model overfits A LOT!
It's relatively easy to predict Gesture / Non gesture
History window of 15 steps is enough for the model to catch up
Leaky std means normalization by subject helps a lot - so there are some subject's device differences (calibration / demographics data?).
ROC AUC very close to 1 - 0.998+ - so we need more data - augmentation for sure (can we generate our own data? or just augment what we have? or predict not on sequence level but on point-wise level N-predictions looking to the past?)
Looked at a competition prediction process and tried to submit lgbm very basic predictions

CMI - Baseline v0
Succeeded · 2mo ago 
0.704 / 0.722
Ok it works. Think dude, think, - do I have something to improve in the process (not talking about the scores - was obvious that it's not even close to be good).
I can use Different models for each target - 1 for multiclass and one for binary.

[ADDed to plan] - Make two different models and use both if inference time and memory permit so (and if binary model would be better)
LGBM still but with multiclass problems - I thought to switch to catboost or xgb, but it was obvious that gradient boosting is a dead-end.

I kept lgbm flow to test model reaction to FE to decide what to keep and what to extend (was useless in the end as no matter how good the feature was, NN treats it differently, and
almost every custom feature made NN results worse)

Also, was thinking to do PCA / Autoencoder for gradient boosting but decided to switch to pure NN.

TARGETs | What we are predicting?
Ok, we have a baseline with almost no feature engineering; it's time to go deeeeeper.

Let's check what our targets are.

df.drop_duplicates(subset=['sequence_id'])['gesture'].value_counts()
gesture
Forehead - pull hairline                      640
Neck - pinch skin                             640
Text on phone                                 640
Neck - scratch                                640
Forehead - scratch                            640
Eyelash - pull hair                           640
Above ear - pull hair                         638
Eyebrow - pull hair                           638
Cheek - pinch skin                            637
Wave hello                                    478
Write name in air                             477
Pull air toward your face                     477
Feel around in tray and pull out an object    161
Write name on leg                             161
Pinch knee/leg skin                           161
Scratch knee/leg skin                         161
Drink from bottle/cup                         161
Glasses on/off                                161
Name: count, dtype: int64
Stop WHAT? Is each person doing exactly the same gestures?

SUBJ_000206  Above ear - pull hair                         8
             Cheek - pinch skin                            8
             Eyebrow - pull hair                           8
             Eyelash - pull hair                           8
             Forehead - pull hairline                      8
             Forehead - scratch                            8
             Neck - pinch skin                             8
             Neck - scratch                                8
             Text on phone                                 8
             Pull air toward your face                     6
             Wave hello                                    6
             Write name in air                             6
             Drink from bottle/cup                         2
             Feel around in tray and pull out an object    2
             Glasses on/off                                2
             Pinch knee/leg skin                           2
             Scratch knee/leg skin                         2
             Write name on leg                             2
SUBJ_001430  Above ear - pull hair                         8
             Cheek - pinch skin                            8
             Eyebrow - pull hair                           8
             Eyelash - pull hair                           8
             Forehead - pull hairline                      8
             Forehead - scratch                            8
             Neck - pinch skin                             8
             Neck - scratch                                8
             Text on phone                                 8
             Pull air toward your face                     6
             Wave hello                                    6
             Write name in air                             6
             Drink from bottle/cup                         2
             Feel around in tray and pull out an object    2
             Glasses on/off                                2
             Pinch knee/leg skin                           2
             Scratch knee/leg skin                         2
             Write name on leg                             2
Ok - THIS is will be our Post-processing! - don't know what exactly and how, but here is the key!

NN experiments
Tried different NNs configurations (thanks to Junie it was fast). Decided to go with different 'branches' for each modality and customizable heads.

Junie talks here:

### Model Overview

Our solution is a multi‑modal neural network tailored for gesture recognition from heterogeneous time‑series features. The architecture is assembled dynamically from a configuration: for each feature group (e.g., raw IMU, gravity‑removed signals, engineered features, ToF‑derived 2D maps), the model instantiates a dedicated encoder and fixes its
output dimension for later fusion. The encoded features are concatenated and passed through two independent fusion blocks with modality dropout. The first fusion stream feeds the main multiclass classifier over all gesture and non‑gesture categories, while the second stream feeds an auxiliary binary detector (gesture vs non‑gesture). Optionally, the second stream can also produce a small auxiliary multiclass head and a lightweight regression head .

### Modality‑Specific Encoders

Each active feature group uses a processor chosen by configuration. Supported processors include temporal encoders for 1D sequences, a compact `dense` processor, and several ToF 2D backbones. Only features with `use_feat=True` are instantiated. Every encoder projects inputs to a fixed `output_dim`, and these embeddings are concatenated before fusion. This modular design allows easy ablations or expansions of the feature set without changing the training loop.

### Fusion and Classification Heads

We use a unified fusion module with configurable width and dropout, plus modality dropout that stochastically drops entire modality embeddings to improve robustness. Both fusion modules receive the same concatenated representation but specialize during training: the first targets fine‑grained multiclass discrimination, and the second refines a clean binary gesture signal. The main head outputs logits over all classes; the auxiliary head outputs a single sigmoid probability for gesture vs non‑gesture. When enabled, the second stream can also drive an auxiliary multiclass classifier and a simple regression head to act as inductive regularizers.

### Training Objectives and Optimization

Training couples the main multiclass objective with the auxiliary binary objective using a family of combined losses selected by configuration with optional mixup support. When auxiliary heads are active, we add a cross‑entropy term for the auxiliary multiclass target and an MSE term for the regression head, both with small weights. The loop supports per‑batch learning‑rate scheduling, optional EMA of model weights, and an optional dropout annealer to gradually adjust regularization. This setup stabilizes optimization across long sequences and heterogeneous modalities.

### Inference and Evaluation

At inference, the model encodes each active feature group, concatenates embeddings, and runs both fusion pathways to produce predictions. For validation, we compute metrics aligned with the competition goals: binary ROC AUC and F1 for gesture vs non‑gesture; mean ROC AUC across the first eight gesture classes; macro‑F1 over the full label space; and a clipped gesture‑only macro‑F1. The reported competition metric is the average of the binary F1 and the gesture macro‑F1, balancing detection quality with class discrimination. A simple TTA mode can aggregate predictions at the sequence level for additional robustness.

### Practical Advantages

- Modular encoders let us integrate diverse signals and swap architectures per modality without touching the core training loop.
- Dual fusion streams decouple fine‑grained classification from robust gesture detection, improving stability and overall performance.
- Auxiliary heads provide targeted regularization and richer supervision.
- Modality dropout enhances resilience when some feature channels are missing or noisy.
For the code agent (Junie) I've created a special folder and some instructions. It will be his personal sandbox to play around and not 'make sh.t files' in other places.

## Experiments Structure and Organization

The `src/experiments` directory is organized to track model experiments:

- `proposal/`: Contains experiment proposals in markdown format
    - Each file represents a single experiment proposal
    - Naming convention: `exp_XXX_short_description.md` where XXX is a sequential number

- `running/`: Contains experiments that are currently in progress
    - Experiments moved here from `proposal/` when they are being implemented
    - May include additional files related to the experiment implementation

- `results/`: Contains completed experiment results
    - Final outcomes, metrics, and analysis of experiments
    - Should reference the original proposal

Each experiment proposal follows a standardized format with sections for:

- Objective
- Hypothesis
- Methodology
- Evaluation Metrics
- Expected Outcome
- Resources Required
- References

The experiment workflow involves:

1. Creating a proposal in the `proposal/` directory
2. Moving it to `running/` during implementation
3. Documenting results in the `results/` directory
4. Using findings to inform future experiments
In the end I came to such modalities and features (features are added / turned off / turned on in different models):

# 1d Temporal
raw_acc = [
    'raw__|acc_x', 'raw__|acc_y', 'raw__|acc_z', 'raw__|acc_magnitude_all',
    'raw__|acc_x__dif__|lag-1', 'raw__|acc_y__dif__|lag-1', 'raw__|acc_z__dif__|lag-1', 'raw__|acc_magnitude_all__dif__|lag-1',
    'raw__|acc_x__ratio__|lag-1', 'raw__|acc_y__ratio__|lag-1', 'raw__|acc_z__ratio__|lag-1', 'raw__|acc_magnitude_all__ratio__|lag-1',
]

# 1d Temporal
raw_rot = [
    'raw__|rot_x', 'raw__|rot_y', 'raw__|rot_z', 'raw__|rot_w',
    'raw__|rot_x__dif__|lag-1', 'raw__|rot_y__dif__|lag-1', 'raw__|rot_z__dif__|lag-1', 'raw__|rot_w__dif__|lag-1',
]

# 1d Temporal
raw_thm = [
    'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',
    'thm__global__|mean', 'thm__global__|std', 'thm__global__|min', 'thm__global__|max',
    'tof__global__|mean', 'tof__global__|std', 'tof__global__|min', 'tof__global__|max', 
]

# 2d Temporal
raw_tof = [c for c in list(base_df) if 'tof_v' in c]

# 1d Temporal
raw_vel = [
    'ang__|velocity_acc_x', 'ang__|velocity_acc_y', 'ang__|velocity_acc_z',
    'ang__|velocity_acc_x__dif__|lag-1','ang__|velocity_acc_y__dif__|lag-1', 'ang__|velocity_acc_z__dif__|lag-1',
]

raw_custom = [
    'raw__|acc_magnitude_all', 'raw__|acc_magnitude_xy', 'raw__|acc_magnitude_zx', 'raw__|acc_magnitude_zy',
    'raw__|rot_angle', 'raw__|rot_angle__dif__|lag-1',
    'raw__|acc_xy_angle', 'raw__|acc_zx_angle', 'raw__|acc_zy_angle',
    'no_grav__|acc_magnitude_all', 'no_grav__|acc_magnitude_all__dif__|lag-1',
    ...
]

# Dense
raw_global = [
    'raw__|acc_x__global__|std', 'raw__|acc_y__global__|std', 'raw__|acc_z__global__|std',
    'raw__|rot_w__global__|std', 'raw__|rot_x__global__|std', 'raw__|rot_y__global__|std', 'raw__|rot_z__global__|std',
    ...
]

# Dense

raw_demo = [
    'adult_child', 'sex', 'handedness', 'age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm',
]
FE and data cleaning
I've tried different approaches like flipping hand / small rotation / TTA - spent too much time with it without success.
Simple things were left as base:

left handed flip 'acc_x' 'rot_y' 'rot_z'] | *-1
wierd 'SUBJ_045235', 'SUBJ_019262' where acc_y seems to be flipped - *-1 slightly improved cv
FE

Gravity removal - 2 versions
Velocity
Distance
Angles and projections
Magnitudes **2 / **3 all and just pairs
Lags / Difs / Ratios
No fancy staff here

Added custom naming for consistency across different versions like 'raw__|rot_x__dif__|lag-1' - was simple to add new features and experiment
Pause and Return
11 of August I had local validation 0.805- 0.815 for IMU data and stopped working on competition after 3 weeks. Completely stopped kaggling as I started new work at JetBrains and all time was dedicated to it (another reason was that I thought that my model is not good at all for gold zone).

Friday night 29 August - good Portuguese wine in my glass, working day came to an end / 20.00 time / no hobby - of course let's check what I did for competition and review with clean eyes.

WHAT!??? Error? Huge error on competition metric. I was experimenting on TTA, and my final metric was a bit weird (forgot to change back to normal)))

Simplification example:
f1 on
mask = binary_predictions > 0.5
np.clip(np.argmax(values[mask], axis=1), 0, 7)
+
f1 on
np.argmax(values, axis=1)<=7

/ 2

Of course, it's not a competition metric - immediately changed to np.clip(np.argmax(.values, axis=1), 0, 8)
and tadaaam I have 0.82-0.824.
So I have a competitive single model - THM and ToF should give + 0.03
0.85 + ensembl - 0.86 + PP 0.87 - I can challenge the gold zone

The config
During the weekend I rewrote all my models and prepared unified config to start making ensemble

ALWAYS use Tensorboard (or equivalent) - because overwise you'll not be able to make good comparisons.

Unified config looks like this

base_params = {
    'GLOBAL_VERBOSE': False,

    # Dataset Params
    'WINDOW_SIZE': 140,
    'OUT_WINDOW_SIZE': 386,  
    'CUR_FOLD': 0,
    'TRAIN_OFFSET': 0,
    'MAIN_TARGET': 'target_b', # MAIN multiclass
    'AUX_TARGET': 'target_a', # Aux binary
    'AUX_TARGET_WEIGHT': 0.3,

    'AUX_TARGET_2': 'target_c', # Aux multiclass
    'AUX2_NUM_CLASSES': 4,

    'AUX_TARGET_3': True, # Regression target
    'AUX3_WEIGHT': 0.05,

    'FEATURES': {}, # added on fly
    'DIMS': {}, # added on fly
    'NUM_CLASSES': 18,
    'FBFILL': False,
    'FILLNA_VALUE': 0,
    'USE_SCALER': True,

    # Batch params
    'BATCH_SIZE': 64,
    'EVAL_BATCH_SIZE': 256 * 8,

    # Feature Engineering
    'MAKE_FE': False, # Runtime fe

    # Model params
    'LEARNING_RATE': 0.0005, 
    'NUM_EPOCHS': 30, 
    'WEIGHT_DECAY': 0.0001,
    'EMA_DECAY': 0.999,

    # Scheduler params
    'SCHEDULER': 'cosine_warmup_decay',
    'WARMUP_EPOCHS': 5, 
    'WARMUP_LR': 0.0001,
    'MIN_LR': 0.00008,

    # Augmentations
    'USE_TTA': False,
    'DS_AUG': 10,
    'MIXUP': True,
    'MIXUP_PROB': 0.8,
    'MIXUP_ALFA': 0.5,
    'RAW_AUG': {},
    'TS_AUG': {
            'enable': True,
            'apply_to': ['raw_acc', 'raw_rot', 'raw_vel', 'raw_thm', 'raw_tof', 'raw_custom'],
            'time_stretch_range': (0.8, 1.2),
            'time_shift_range': 0.1,
            'noise_std': 0.02,
            'magnitude_scale_range': (0.9, 1.1),
            'rotation_angle_range': 0.1,
            'mask_ratio': 0.1,
            'freq_filter_range': (0.1, 0.9),
            'p_stretch': 0.,
            'p_shift': 0.,
            'p_noise': 0.5,
            'p_mag': 0.1,
            'p_rotate': 0.,
            'p_mask': 0.1,
            'p_freq': 0.,
        },

    # Other
    'SEED': 18,

    # losss
    'LOSS_NAME': 'v1',
    'NEGATIVE_PROBS': False,

    # Layers params
    'fusion_channels': 512,
    'fusion_dropout': 0.5,
    'fusion_modality_dropout': 0.2,

    'PROCESSOR_CONFIGS': {
        "raw_acc": {
            "processor_type": "temporal_v2",
            "use_feat": True,
            "output_dim": 512,
            "dropout": 0.5,
            "res_blocks": {
                "block1": {
                    "in_channels": None,
                    "channels": [64, 128],
                    "ks": [3, 5],
                    "dropout": 0.5,
                    "reduction": 8,
                    "pool_size": 2,
                    "core_type": "cnn2"
                },
                "block2": {
                    "in_channels": 256,
                    "channels": [128, 256, 256],
                    "ks": [3, 5, 5],
                    "dropout": 0.5,
                    "reduction": 8,
                    "pool_size": 2,
                    "core_type": "cnn3"
                }
            }
        },
        ...
        'raw_tof': {
            'use_feat': True,
            'processor_type': 'tof_conv2d',
            'output_dim': 64,
            'conv_channels': [8, 16, 32],
            'kernel_size': 3,
            'dropout': 0.5,
            'pool_size': 2,
            'temporal_dim': 512
        },
        'raw_global': {
            'processor_type': 'dense',
            'use_feat': False,
            'output_dim': 32,
            'hidden_dim': 64,
            'dropout': 0.7,
            'use_bn': True,
        },

    # Dropout shdeuler
    'USE_DROPOUT_SCHEDULER': False,
}
Such a config permitted changing params on the fly and track any changes and rerun model inference on kaggle with stable results.

Good models were uploaded to the Kaggle dataset, and I made several submissions. Monday - Thursday I slowly trained IMU models as didn't 100% believe in success.
25 August night submission of 9 / 5 folded IMU models gave 0.838 on LB.

Junie talks here:

### TemporalProcessor

`TemporalProcessor` is a compact sequence encoder that combines convolutional feature extraction, temporal modeling, and attention-based pooling to produce a fixed-size representation per sequence. It takes an input tensor shaped `[B, T, F]` (batch, time, features), extracts local temporal patterns with residual 1D CNNs, models longer-range dependencies with a bidirectional GRU, and compresses the sequence to a single vector via learnable attention before projecting to a fixed `output_dim`.

#### Convolutional front-end (local pattern extraction)
- Two residual CNN blocks (`ResidualCNNBlock`) operate on the transposed signal `[B, F, T]`:
  - Block 1 adapts channel width from `in_features` and uses a 3-layer or 2-layer 1D CNN core.
  - Block 2 continues feature refinement with its own core.
- Each block includes:
  - Channel attention via a Squeeze-and-Excitation `SEBlock` to reweight feature maps.
  - Residual addition, ReLU activation, `MaxPool1d` downsampling, and dropout.
- This stage increases the receptive field and reduces sequence length (two pooling steps) while maintaining stable gradients through residual connections.

#### Temporal backbone (sequence modeling)
- The convolved output is transposed back to `[B, T', C]` and fed to a bidirectional GRU:
  - `input_size = C` equals the final CNN channel count (from Block 2).
  - `hidden_size = C`, `bidirectional=True`, `batch_first=True`, `bias=False`.
  - A dropout layer follows the GRU to regularize temporal representations.

#### Attention pooling and projection
- A learnable `AttentionLayer` consumes the GRU outputs `[B, T', 2C]` and returns a weighted context vector `[B, 2C]`, focusing the model on the most informative time steps.
- A final linear layer maps this vector to `output_dim`, yielding a per-sequence embedding suitable for downstream fusion and classification.

#### Configurability and I/O
- Key arguments:
  - `in_features`: number of input channels per time step.
  - `res_blocks`: dictionary that configures channels, kernel sizes, pooling, dropout, SE reduction, and `core_type` per block. Defaults provide a strong general-purpose setup:
    - Block 1: `channels=[64,128]`, `ks=[3,5]`, `core_type='cnn2'`.
    - Block 2: `channels=[128,128,256]`, `ks=[3,5,5]`, `core_type='cnn3'`.
  - `dropout`: GRU output dropout; each block has its own dropout.
  - `output_dim`: final embedding size exported to the fusion module.
- Shapes:
  - Input: `[B, T, in_features]`
  - After CNN blocks: downsampled in time, expanded in channels
  - After BiGRU: `[B, T', 2C]`
  - Output: `[B, output_dim]` (sequence-level vector)

#### Design rationale
- Residual CNNs with SE attention capture multi-scale local dynamics and suppress noisy channels early.
- Max pooling reduces computational load and encourages invariance to minor timing shifts.
- The bidirectional GRU complements CNNs by modeling longer contexts beyond the convolutional receptive field.
- Attention pooling provides a flexible, data-driven way to summarize variable-length sequences.
- The fixed `output_dim` makes this processor plug-and-play within the multi-modal fusion framework, allowing different feature groups to contribute comparably sized embeddings.
Last Days
Ok, now I had and option or FULL IN or just STOP. Danila Savenkov (JB side, you may know him as @daniel89) permitted me to spend 3 days on Kaggle (Friday / Monday / Tuesday + Weekend, of course).

Decision made - FULL IN.
What I need:

GPU - lighting AI - 1 own gpu is not enough
ToF model
Postprocessing
Better IMU model
LUCK)))
Carefull LB probing
Not f.ck up with CV as had no chance to correct anything
Thursday Night / Friday
LB probing with ensemble setup - average / weighted average / number models / full data or 5 folds - 0.838 still
Decided to go with many week 5 folded models - this was the only chance as had no option to probe separate single model on full data
Let's make ToF model - 0.855 with IMU mix wih 1 ToF
I need more ToF models and more IMU - started mass training
Satruday
I need post-process - NO way 1st place has such a difference in score with models only
First PP version - 0.862 - a good start, but I need more
More ToF models - 5 ready and validated - 0.871 with PP (no submissions left to probe without)
Fixed EMA model - was cloning trainable param but not buffers - fix made EMA works
Sunday
Switch to EMA only models

Changed LR scheduler

Retrained IMU Models

Made Massive probing (to make sure all components work as intended)

Single IMU only no PP - (0.823 / 0.834)

6 IMU models no PP weighted average - (0.837 / 0.845)

6 IMU models with PP weighted average - (0.840 / 0.853)

6 IMU + old ToF no PP (0.851 / 0.868)

6 IMU + old ToF with PP (0.863 / 0.876)

Did few changes to Post-processing but didn't manage to improve much the output my local CV showed +0.01 +/- 0.0015 std / max-min 0.006

Monday
New ToF models
no PP (0.853 / 0.872)
with PP (0.868 / 0.883)
Tuesday - Final day
Timing should be very precise
All new models scores should be ready till 22.00 London time to submit a final solution
New Imu models 6 runs - 1.5H each on 3 gpu should be ready till 16 00 - 2h inference - 19 00 scores maximum
New ToF models - 3h each run - will be able to make 3 models more - should be ready till 19 00 - 21 00 scores maximum
1h to prepare final mix with pp and submit till 22 00
CMI - Final 5.0 - Final candidate 2!
Succeeded · Tue Sep 02 2025 21:58:40 GMT+0100 (0.860 / 0.877)

CMI - Final 4.0 - Final Candidate 1!
Succeeded · Tue Sep 02 2025 21:57:42 GMT+0100 (0.858 / 0.880)

Done - choose Candidate 1 and previous best lb model - probably 0.858 would be enough for gold also but we will never know)))
Models changes experiments
features
different CNN block - 'cnn', 'inception', 'gated_dilated', 'convnext', 'miniconformer', 'fft'
conv layers
output dims
extrapolated window
LR
dropouts
fusion blocks
aux targets
loss functions
Had no option to test if my weights are right - probably a simple average mix with cv score cutoff would perform better.

Mix was made with Dirichlet search with oof predictions (too risky if CV is wrong and not aligned with LB) - cutoff low weights and normalized back

IMU Final metric (rounded+pruned): 0.843523 | 0.854 with pp
TOF Final metric (rounded+pruned): 0.891330 | 0.903 with pp
#################################
# Models Weights
#################################

# IMU
'test_run_-_2025-09-02_102121': np.float64(0.04577114427860696),
 'test_run_-_2025-09-02_104733': np.float64(0.07462686567164178),
 'test_run_-_2025-09-02_100513': np.float64(0.13532338308457711),
 'test_run_-_2025-09-02_100715': np.float64(0.04676616915422885),
 'test_run_-_2025-09-01_232203': np.float64(0.07462686567164178),
 'test_run_-_2025-08-31_182223': np.float64(0.16417910447761194),
 'test_run_-_2025-08-31_182238': np.float64(0.04577114427860696),
 'test_run_-_2025-08-31_182155': np.float64(0.055721393034825865),
 'test_run_-_2025-08-30_144324': np.float64(0.04577114427860696),
 'test_run_-_2025-08-26_125131': np.float64(0.04676616915422885),
 'test_run_-_2025-08-26_095603': np.float64(0.1024875621890547),
 'test_run_-_2025-08-26_055724': np.float64(0.05771144278606965),
 'test_run_-_2025-08-26_055618': np.float64(0.10447761194029849)
}

# TOF
{'test_run_-_2025-09-01_174406': np.float64(0.08208884884305259),
 'test_run_-_2025-09-01_095855': np.float64(0.027909755890227466),
 'test_run_-_2025-09-01_022627': np.float64(0.15450960916030498),
 'test_run_-_2025-08-30_112546': np.float64(0.029019711064357332),
 'test_run_-_2025-08-29_151211': np.float64(0.028584738288600937),
 'test_run_-_2025-08-28_124723': np.float64(0.009465796363042244),
 'test_run_-_2025-08-28_124649': np.float64(0.07006594305155826),
 'test_run_-_2025-08-29_091104': np.float64(0.08299200257345174),
 'test_run_-_2025-08-28_154159': np.float64(0.13787604157789557),
 'test_run_-_2025-08-27_205748': np.float64(0.028090284333857785),
 'test_run_-_2025-08-27_141837': np.float64(0.12763658458693833),
 'test_run_-_2025-08-31_182223': np.float64(0.11118294788297756),
 'test_run_-_2025-08-31_182142': np.float64(0.005057595225800813),
 'test_run_-_2025-08-31_182238': np.float64(0.06738590961562396),
 'test_run_-_2025-08-31_182155': np.float64(0.00539168026876913),
 'test_run_-_2025-08-26_170459': np.float64(0.010176717301051534),
 'test_run_-_2025-08-26_095603': np.float64(0.010695373071077567),
 'test_run_-_2025-08-26_055724': np.float64(0.01187046090141201)
}
After words.
I joined Kaggle almost 7 years ago. Never thought that I would be a triple Grandmaster - till today I had the shame even to call myself a GM as only competition title matters.
The road to the last medal was long))) It's not the end - I'll compete (when will have more free time) and definitely will compete always when tabular data appears (not stock market)).
So many good memories come to me:

My first competitions - top 50 places when I believed that I could make some good models
First Team with @simpletonwang and @xkagjaeo05 (still friends btw) and how we missed our gold (due to bad time management)
1st place in IEEE Fraud competition - with one and only @cdeotte https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/111284 https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
Dream team with @johnpateha @headsortails @cdeotte and our fail on M5 competition (with guessing distribution on private test part) - https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/163621
AMEX competition - how kaggle noobs took 2nd place)))
1st place with @dott1718 on AMP Parkinson's Disease Progression Prediction (when I had a privilege to work with a pure genius)
@trasibuloon first CMI competition and how we were trying to solve metric puzzle))
Emotional discussions / I would call them FIGHTS)))) with @bakeryproducts (we met and worked together because of Kaggle), who taught me A LOT!
I found so many friends because of Kaggle - not only during competitions but also social networks, meetups, conferences!
Without you guys - your mentoring, long conversations, your example, I wouldn't achieve all these.
And of course the Kaggle community - the support, knowledge sharing, gentle teaching - there is no other place like this.

THANK YOU!


Vladimir Demidov
Posted 10 months ago

· 250th in this Competition

Really admire your style of storytelling - it’s one of the best write-ups I’ve seen here. The way you share your journey from your own perspective, with such depth and empathy, makes readers feel as if they’re walking your path alongside you. Even the technical parts about NNs, paired with Junie’s description, come across as almost poetic.


Reply

3
Fnoa
Posted 10 months ago

· 9th in this Competition

Great write-up, Konstantin. I found it very engaging to read. I’m so glad we connected through the first CMI competition, and I really admire your skills with tabular data.


Reply

1
Konstantin Yakovlev
Topic Author
Posted 10 months ago

· 8th in this Competition

To keep all in one place

Post-processing
https://www.kaggle.com/code/kyakovlev/cmi-8th-place-postprocessing


Reply

React
MJeremy
Posted 10 months ago

· 293rd in this Competition

hi man, it's a touchy read. Only people finish the read can relate.

btw, how do you organise all the changes in like an IDE format? did you run the code locally or you have your local setup connect to Kaggle VM? How does the whole setup work to allow fast training and experiment at the same time


Reply

React
Konstantin Yakovlev
Topic Author
Posted 10 months ago

· 8th in this Competition

I have a single local GPU and majority of changes were tested on single fold locally (in the notebook). Also had end to end "pipeline" but notebooks are more convenient for me.

For bulk experiments I've used lightning.ai where I clone "src" folder and notebook that accepts config (tensorboard logs, oofs and weights downloaded to local machine for evaluation and comparison)

candidate_name = f'test_run_-_{datetime_str}'
tb_dir = f'{candidate_dir}/tb'


writer = SummaryWriter(log_dir=f'{tb_dir}/seed-{SEED}__fold-{fold_idx}')  

for k, v in validation_results.items():
    if isinstance(v, (int, float)):
         writer.add_scalar(f'Val/{k}', v, epoch)
# Naming is terrible I know)))
validation_results = {
        'loss': val_loss,
        'f1': gesture_f1,
        'roc_auc': gesture_roc_auc,
        'f1_mean': gesture_f1_mean,
        'roc_auc_mean': gesture_roc_auc_mean,

        'g8_roc_auc': g8_mean_roc_auc,

        'g8_f1': g8_gesture_f1,
        'g8_f1_clipped': g8_gesture_f1_clipped,
        'g8_auc': mean_match,
        'comp_metric': competition_metric,
    }

Reply

React

6 more replies
Profile picture for MJeremy
Profile picture for Konstantin Yakovlev
Konstantin Yakovlev
Topic Author
Posted 10 months ago

· 8th in this Competition

And here is a customizable model:

class GestureModel(nn.Module):
    def __init__(self, all_params):
        super(GestureModel, self).__init__()

        # Store params
        self.all_params = all_params
        self.num_classes = all_params['NUM_CLASSES']

        # Extract feature configurations
        self.features = all_params['FEATURES']
        self.dims = all_params['DIMS']

        # Initialize processors dynamically
        self.processors = nn.ModuleDict()
        self.processor_out_dims = {}

        # Create processors for each feature type
        for feat_name, feat_data in self.features.items():
            if feat_data is not None and len(feat_data) > 0:
                dim_key = f'{feat_name}_dim'
                if dim_key in self.dims:
                    self._create_processor(feat_name, self.dims[dim_key])

        # Fusion
        self._create_fusion_processors(all_params)

        # Classifiers
        self.classifier = nn.Linear(self.fusion_1.output_dim, self.num_classes)
        self.binary_classifier = nn.Linear(self.fusion_2.output_dim, 1)

        # Optional multiclass auxiliary head
        self.aux2_target = all_params.get('AUX_TARGET_2', None)
        self.aux2_num_classes = all_params.get('AUX2_NUM_CLASSES', 4)
        self.use_aux2 = self.aux2_target is not None
        if self.use_aux2:
            self.aux2_classifier = nn.Linear(self.fusion_2.output_dim, self.aux2_num_classes)

        # Optional target_e mean auxiliary head
        self.aux3_target = all_params.get('AUX_TARGET_3', False)
        self.use_aux3 = self.aux3_target
        if self.use_aux3:
            self.aux3_classifier = nn.Linear(self.fusion_2.output_dim, 1)

    def _create_fusion_processors(self, all_params):
        """Create fusion processors with dynamic dimensions."""

        # Get fusion parameters
        self.fusion_channels = all_params.get('fusion_channels', 512)
        self.fusion_dropout = all_params.get('fusion_dropout', 0.4)
        self.fusion_modality_dropout = all_params.get('fusion_modality_dropout', 0.2)

        # Calculate total input dimension and split dimensions dynamically
        active_features = []
        split_dims = []

        for feat_name in self.features:
            if feat_name in self.processor_out_dims:
                active_features.append(feat_name)
                split_dims.append(self.processor_out_dims[feat_name])

        total_input_dim = sum(split_dims)

        if total_input_dim == 0:
            raise ValueError("No active features found for fusion processors")

        # Create fusion processors with dynamic dimensions
        self.fusion_1 = FusionProcessorNew(
            input_dim=total_input_dim,
            fusion_channels=self.fusion_channels,
            dropout=self.fusion_dropout,
            split_dims=tuple(split_dims),
            modality_dropout=self.fusion_modality_dropout
        )

        self.fusion_2 = FusionProcessorNew(
            input_dim=total_input_dim,
            fusion_channels=self.fusion_channels,
            dropout=self.fusion_dropout,
            split_dims=tuple(split_dims),
            modality_dropout=self.fusion_modality_dropout
        )

    def _create_processor(self, feat_name, input_dim):
        """Create processor based on feature type and dimensions."""

        # Get processor configurations from all_params
        processor_configs = self.all_params.get('PROCESSOR_CONFIGS', {})

        # Find matching configuration
        config = None
        for config_key in processor_configs:
            if config_key in feat_name:
                config = processor_configs[config_key].copy()
                break

        # Use default configuration if no match found
        if config is None:
            print('### Error: no processor config for {}'.format(feat_name))
            return

        if not config.get('use_feat', False):
            print(f'### Skipping processor creation for {feat_name} (use_feat=False)')
            return

        print('Created processor config for {}'.format(feat_name))

        # Remove use_feat from config before passing to TemporalProcessor
        config_for_processor = config.copy()
        config_for_processor.pop('use_feat', None)

        if config_for_processor['processor_type'] == 'temporal':
            config_for_processor.pop('processor_type', None)
            # Create processor
            self.processors[feat_name] = TemporalProcessor(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'temporal_v2':
            config_for_processor.pop('processor_type', None)
            # Create processor
            self.processors[feat_name] = TemporalProcessor_v2(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'triple_temporal':
            config_for_processor.pop('processor_type', None)
            self.processors[feat_name] = TripleTemporalProcessor(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'dense':
            config_for_processor.pop('processor_type', None)
            self.processors[feat_name] = DenseProcessor(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'tof_conv2d':
            config_for_processor.pop('processor_type', None)
            self.processors[feat_name] = TofConv2DProcessor(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'tof2d_sensor_channels':
            config_for_processor.pop('processor_type', None)
            self.processors[feat_name] = TofConv2DProcessor_SensorAsChannel(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'tof2d_inception_cbam':
            config_for_processor.pop('processor_type', None)
            self.processors[feat_name] = TofConv2DProcessor_InceptionCBAM(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'tof2d_axial_tcn':
            config_for_processor.pop('processor_type', None)
            self.processors[feat_name] = TofConv2DProcessor_AxialTCN(
                in_features=input_dim,
                **config_for_processor
            )
        elif config_for_processor['processor_type'] == 'tof2d_cross_attn':
            config_for_processor.pop('processor_type', None)
            self.processors[feat_name] = TofConv2DProcessor_CrossSensorAttn(
                in_features=input_dim,
                **config_for_processor
            )

        self.processor_out_dims[feat_name] = config['output_dim']

    def forward(self, features_dict):

        processed_features = []

        # Process features
        for feat_name in self.features:
            feat_key = f'{feat_name}_features'
            if feat_key in features_dict and feat_name in self.processors:
                feat_tensor = features_dict[feat_key]
                processed_feat = self.processors[feat_name](feat_tensor)
                processed_features.append(processed_feat)

        # Concatenate all processed features
        if processed_features:
            combined_features = torch.cat(processed_features, dim=1)
        else:
            raise ValueError("No valid features found for processing")

        # Fusion and classification
        fused_features_1 = self.fusion_1(combined_features)
        fused_features_2 = self.fusion_2(combined_features)

        class_logits = self.classifier(fused_features_1)
        binary_aux = torch.sigmoid(self.binary_classifier(fused_features_2)).squeeze(-1)

        if getattr(self, 'use_aux2', False) and getattr(self, 'use_aux3', False):
            aux2_logits = self.aux2_classifier(fused_features_2)
            aux3_output = self.aux3_classifier(fused_features_2).squeeze(-1)  # Regression output
            return class_logits, binary_aux, aux2_logits, aux3_output
        elif getattr(self, 'use_aux2', False):
            aux2_logits = self.aux2_classifier(fused_features_2)
            return class_logits, binary_aux, aux2_logits
        elif getattr(self, 'use_aux3', False):
            aux3_output = self.aux3_classifier(fused_features_2).squeeze(-1)
            return class_logits, binary_aux, aux3_output
        else:
            return class_logits, binary_aux

Reply

React
Konstantin Yakovlev
Topic Author
Posted 10 months ago

· 8th in this Competition

Dataset and model both accepts unified config - all configuration is done in a single place.

Customizable Temporal block look like this:

all params comes from config and permits do many experiments without changing NN at all
'in_channels': 128,
'channels': [128, 256, 256],
'ks': [3, 5, 5],
'dropout': 0.3,
'reduction': 16,
'pool_size': 2,
'core_type': 'cnn3',

class ResidualCNNBlock_v2(nn.Module):
    """Residual block; поддержаны 'cnn2' и 'cnn3'."""

    def __init__(
            self,
            in_channels=10,
            channels=None,
            ks=None,
            dropout=0.5,
            reduction=16,
            pool_size=2,
            core_type='cnn3',
    ):
        super().__init__()
        self.core_type = core_type

        if channels is None:
            channels = [64, 128, 256]
        if ks is None:
            ks = [3, 5, 7]

        if core_type == 'cnn2':
            self.core = nn.Sequential(
                nn.Conv1d(in_channels, channels[0], ks[0], padding=ks[0] // 2, bias=False),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(inplace=True),

                nn.Conv1d(channels[0], channels[1], ks[1], padding=ks[1] // 2, bias=False),
                nn.BatchNorm1d(channels[1]),
                nn.ReLU(inplace=True),
            )
            out_channels = channels[1]

        elif core_type == 'cnn3':
            self.core = nn.Sequential(
                nn.Conv1d(in_channels, channels[0], ks[0], padding=ks[0] // 2, bias=False),
                nn.BatchNorm1d(channels[0]),
                nn.ReLU(inplace=True),

                nn.Conv1d(channels[0], channels[1], ks[1], padding=ks[1] // 2, bias=False),
                nn.BatchNorm1d(channels[1]),
                nn.ReLU(inplace=True),

                nn.Conv1d(channels[1], channels[2], ks[2], padding=ks[2] // 2, bias=False),
                nn.BatchNorm1d(channels[2]),
                nn.ReLU(inplace=True),
            )
            out_channels = channels[2]

        else:
            raise ValueError(f'Unknown core_type: {core_type}')

        self.channel_attention = SEBlock(out_channels, reduction=reduction)

        self.shortcut_connection = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.pooling = nn.MaxPool1d(pool_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut_connection(x)
        out = self.core(x)
        out = self.channel_attention(out)
        out = out + residual
        out = self.act(out)
        out = self.pooling(out)
        out = self.dropout_layer(out)
        return out


class TemporalProcessor_v2(nn.Module):
    def __init__(self, in_features=7,
                 res_blocks=None,
                 output_dim=8,
                 dropout=0.2,
                 ):
        super().__init__()

        if res_blocks is None:
            res_blocks = {
                'block1': {
                    'channels': [64, 128],
                    'ks': [3, 5],
                    'dropout': 0.5,
                    'reduction': 8,
                    'pool_size': 2,
                    'core_type': 'cnn2',
                },
                'block2': {
                    'in_channels': 128,
                    'channels': [128, 256, 256],
                    'ks': [3, 5, 5],
                    'dropout': 0.3,
                    'reduction': 16,
                    'pool_size': 2,
                    'core_type': 'cnn3',
                },
            }

        # In features form outside
        res_blocks['block1']['in_channels'] = in_features

        self.block1 = ResidualCNNBlock_v2(**res_blocks['block1'])
        self.block2 = ResidualCNNBlock_v2(**res_blocks['block2'])

        def _out_channels(cfg):
            ct = cfg.get('core_type', 'cnn2')
            ch = cfg['channels']
            return ch[1] if ct == 'cnn2' else ch[2]

        cnn_out_features = _out_channels(res_blocks['block2'])

        self.temporal = nn.GRU(input_size=cnn_out_features,
                               hidden_size=cnn_out_features,
                               bias=False, bidirectional=True, batch_first=True)
        self.temporal_dropout = nn.Dropout(dropout)

        self.attention = AttentionLayer(cnn_out_features * 2)
        self.output_projection = nn.Linear(cnn_out_features * 2, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.block1(x)
        x = self.block2(x)
        x = x.transpose(1, 2)  # [B, T, F]

        self.temporal.flatten_parameters()
        x, _ = self.temporal(x)
        x = self.temporal_dropout(x)

        x = self.attention(x)
        x = self.output_projection(x)
        return x