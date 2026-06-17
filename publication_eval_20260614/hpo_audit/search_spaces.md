# Observed Optuna Search Space

This file reports parameter values observed in completed Optuna trials. It is an audit artifact, not a substitute for the exact search-space definitions in each tuning script.

## randomforest

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `MAX_DEPTH` | numeric | 6.0 | 32.0 | 8 | - |
| `MAX_FEATURES` | categorical_or_mixed | - | - | 6 | 0.35 ; 0.5 ; 0.7 ; 1.0 ; log2 ; sqrt |
| `MIN_SAMPLES_LEAF` | numeric | 2.0 | 8.0 | 7 | - |
| `MIN_SAMPLES_SPLIT` | numeric | 2.0 | 18.0 | 15 | - |
| `N_ESTIMATORS` | numeric | 400.0 | 1200.0 | 8 | - |
| `WAVEFORM_FEATURE_COUNT` | numeric | 64.0 | 384.0 | 4 | - |

## xgboost

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `COLSAMPLE_BYTREE` | numeric | 0.6038154690171463 | 0.9611477201801439 | 50 | - |
| `LEARNING_RATE` | numeric | 0.005981239188691928 | 0.07876882828901684 | 50 | - |
| `MAX_DEPTH` | numeric | 3.0 | 9.0 | 7 | - |
| `MIN_CHILD_WEIGHT` | numeric | 0.6502312170282649 | 19.962608264826628 | 50 | - |
| `N_ESTIMATORS` | numeric | 750.0 | 5000.0 | 17 | - |
| `REG_ALPHA` | numeric | 1.2163192461131192e-06 | 2.4214375123242835 | 50 | - |
| `REG_LAMBDA` | numeric | 0.0010123565000338239 | 15.593447853576093 | 50 | - |
| `SUBSAMPLE` | numeric | 0.6055432131349704 | 0.9993485976853105 | 50 | - |
| `WAVEFORM_FEATURE_COUNT` | numeric | 64.0 | 384.0 | 4 | - |

## lightgbm

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `COLSAMPLE_BYTREE` | numeric | 0.634267888587052 | 0.9948139055847877 | 50 | - |
| `LEARNING_RATE` | numeric | 0.007021506951292295 | 0.07941692219022867 | 50 | - |
| `MAX_DEPTH` | numeric | -1.0 | 16.0 | 7 | - |
| `MIN_CHILD_SAMPLES` | numeric | 27.0 | 199.0 | 43 | - |
| `NUM_LEAVES` | numeric | 15.0 | 217.0 | 40 | - |
| `N_ESTIMATORS` | numeric | 500.0 | 5000.0 | 15 | - |
| `REG_ALPHA` | numeric | 1.036690363447721e-06 | 1.022815612986707 | 50 | - |
| `REG_LAMBDA` | numeric | 0.0010370466048539265 | 22.075242368970418 | 50 | - |
| `SUBSAMPLE` | numeric | 0.6004765093796335 | 0.9865790287667695 | 50 | - |
| `WAVEFORM_FEATURE_COUNT` | numeric | 64.0 | 384.0 | 4 | - |

## catboost

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `DEPTH` | numeric | 4.0 | 10.0 | 7 | - |
| `ITERATIONS` | numeric | 500.0 | 6000.0 | 21 | - |
| `L2_LEAF_REG` | numeric | 1.004059975597095 | 29.437366963651552 | 50 | - |
| `LEARNING_RATE` | numeric | 0.006316197938720379 | 0.07917008932017806 | 50 | - |
| `SUBSAMPLE` | numeric | 0.6001550625705424 | 0.9976544664717388 | 50 | - |
| `WAVEFORM_FEATURE_COUNT` | numeric | 64.0 | 384.0 | 4 | - |

## mlp

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `BATCH_SIZE` | numeric | 256.0 | 1024.0 | 4 | - |
| `EARLY_STOPPING_PATIENCE` | numeric | 15.0 | 30.0 | 4 | - |
| `LEARNING_RATE` | numeric | 1.3408933722521186e-05 | 0.0029702335828275706 | 30 | - |
| `MLP_BLOCK_COUNT` | numeric | 2.0 | 5.0 | 4 | - |
| `MLP_DROPOUT` | numeric | 0.09275589839587947 | 0.3447718513441057 | 30 | - |
| `MLP_HIDDEN_DIM` | numeric | 128.0 | 512.0 | 5 | - |
| `MLP_HIDDEN_MULT` | numeric | 2.0 | 4.0 | 3 | - |
| `SMOOTH_L1_BETA` | numeric | 0.3 | 2.0 | 5 | - |
| `WAVEFORM_FEATURE_COUNT` | numeric | 64.0 | 384.0 | 4 | - |
| `WEIGHT_DECAY` | numeric | 1.6355292386610658e-06 | 0.000716700296826889 | 30 | - |

## lstm

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `BATCH_SIZE` | numeric | 32.0 | 96.0 | 3 | - |
| `FUSION_BILINEAR_DIM` | numeric | 48.0 | 112.0 | 4 | - |
| `FUSION_OUTPUT_DIM` | numeric | 256.0 | 512.0 | 3 | - |
| `GRAD_CLIP_NORM` | numeric | 0.5 | 1.5 | 4 | - |
| `HEAD_DROPOUT` | numeric | 0.10733160256067056 | 0.2680928743451806 | 20 | - |
| `LEARNING_RATE` | numeric | 3.526117917948776e-05 | 0.00039769297255521914 | 20 | - |
| `LSTM_ATTENTION_DIM` | numeric | 64.0 | 192.0 | 3 | - |
| `LSTM_DROPOUT` | numeric | 0.09820519954991144 | 0.34204549231207704 | 20 | - |
| `LSTM_HIDDEN_DIM` | numeric | 96.0 | 256.0 | 4 | - |
| `LSTM_NUM_LAYERS` | numeric | 1.0 | 3.0 | 3 | - |
| `SCALAR_EMBED_DIM` | numeric | 128.0 | 256.0 | 3 | - |
| `SCALAR_RES_BLOCKS` | numeric | 2.0 | 5.0 | 4 | - |
| `SCALAR_RES_DROPOUT` | numeric | 0.11142334447024602 | 0.2852729964247317 | 20 | - |
| `SEQUENCE_PROJECTOR_DIM` | numeric | 128.0 | 384.0 | 4 | - |
| `SEQ_LEN` | numeric | 1024.0 | 4096.0 | 3 | - |
| `SEQ_STEM_CHANNELS_KEY` | categorical_or_mixed | - | - | 3 | 24-48 ; 48-96 ; 64-128 |
| `SEQ_STEM_DROPOUT` | numeric | 0.03301951846183145 | 0.1685333104604018 | 20 | - |
| `WEIGHT_DECAY` | numeric | 7.998282299351469e-06 | 0.0003832048635270239 | 20 | - |

## wavenet

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `BATCH_SIZE` | numeric | 48.0 | 96.0 | 3 | - |
| `FUSION_BILINEAR_DIM` | numeric | 48.0 | 112.0 | 4 | - |
| `FUSION_OUTPUT_DIM` | numeric | 256.0 | 512.0 | 3 | - |
| `GRAD_CLIP_NORM` | numeric | 0.8 | 1.5 | 3 | - |
| `HEAD_DROPOUT` | numeric | 0.10224209721782068 | 0.3151886489788739 | 20 | - |
| `LEARNING_RATE` | numeric | 5.489721757656989e-05 | 0.00043261568993104704 | 20 | - |
| `SCALAR_EMBED_DIM` | numeric | 128.0 | 256.0 | 3 | - |
| `SCALAR_RES_BLOCKS` | numeric | 2.0 | 5.0 | 4 | - |
| `SCALAR_RES_DROPOUT` | numeric | 0.08198146579569754 | 0.2660654672984202 | 20 | - |
| `SEQUENCE_PROJECTOR_DIM` | numeric | 192.0 | 384.0 | 4 | - |
| `SEQ_LEN` | numeric | 2048.0 | 4096.0 | 2 | - |
| `WAVENET_ATTENTION_DIM` | numeric | 64.0 | 192.0 | 4 | - |
| `WAVENET_DILATIONS_KEY` | categorical_or_mixed | - | - | 3 | 1-2-4-8-16-32 ; 1-2-4-8-16-32-64 ; 1-2-4-8-16-32-64-128-256 |
| `WAVENET_DILATION_CYCLES` | numeric | 2.0 | 4.0 | 3 | - |
| `WAVENET_DROPOUT` | numeric | 0.04332060667501372 | 0.20666873426233 | 20 | - |
| `WAVENET_KERNEL_SIZE` | numeric | 3.0 | 5.0 | 2 | - |
| `WAVENET_RESIDUAL_CHANNELS` | numeric | 32.0 | 96.0 | 4 | - |
| `WAVENET_SKIP_CHANNELS` | numeric | 64.0 | 192.0 | 4 | - |
| `WEIGHT_DECAY` | numeric | 1.0808774559463289e-06 | 9.063980376767424e-05 | 20 | - |

## 2dcnn

| Parameter | Type | Observed min | Observed max | Unique count | Example values |
| --- | --- | ---: | ---: | ---: | --- |
| `BATCH_SIZE` | numeric | 48.0 | 96.0 | 3 | - |
| `CNN_CHANNELS_KEY` | categorical_or_mixed | - | - | 3 | 24-56-112-176 ; 32-72-144-224 ; 40-80-160-256 |
| `CNN_DROPOUT` | numeric | 0.052670212739291976 | 0.18890613585815438 | 20 | - |
| `CNN_PROJECTOR_DIM` | numeric | 192.0 | 384.0 | 4 | - |
| `CNN_PROJECTOR_DROPOUT` | numeric | 0.08216951745101624 | 0.24271501966693187 | 20 | - |
| `FUSION_BILINEAR_DIM` | numeric | 48.0 | 112.0 | 4 | - |
| `FUSION_DROPOUT` | numeric | 0.083288997328408 | 0.27924475141287436 | 20 | - |
| `FUSION_OUTPUT_DIM` | numeric | 256.0 | 512.0 | 3 | - |
| `GRAD_CLIP_NORM` | numeric | 0.5 | 1.5 | 4 | - |
| `HEAD_DROPOUT` | numeric | 0.10705281830545887 | 0.33430837343557707 | 20 | - |
| `IMAGE_SIZE_KEY` | numeric | 128.0 | 192.0 | 3 | - |
| `LEARNING_RATE` | numeric | 3.107507395124893e-05 | 0.0004137763716148568 | 20 | - |
| `SCALAR_EMBED_DIM` | numeric | 128.0 | 256.0 | 3 | - |
| `SCALAR_RES_BLOCKS` | numeric | 2.0 | 5.0 | 4 | - |
| `SCALAR_RES_DROPOUT` | numeric | 0.11106394326487329 | 0.296782606149452 | 20 | - |
| `TIME_FREQ_MASK_PROB` | numeric | 0.11410547178880269 | 0.4382592519882411 | 20 | - |
| `WEIGHT_DECAY` | numeric | 1.0791395220942102e-06 | 0.0004633121220485119 | 20 | - |
