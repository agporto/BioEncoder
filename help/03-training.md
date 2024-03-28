# Training config

BioEncoder relies on _YAML_ files to control the training process. Each _YAML_ file contains several hyperparameters that can be modified according to your needs. These hyperparameters include:

- Model architecture
- Augmentations
- Loss functions
- etc..

Example config files can be found in the [config-templates](../config-templates) folder. These files provide a starting point for training BioEncoder models and can be modified to suit specific use cases.

## Training stage 1

To train stage 1 of the model, run the following command:

```
bioencoder.train(config_path=r"bioencoder_configs/train_stage1.yml")
```

Trace progress with tensorboard:

```
tensorboard --logdir "bioencoder_wd/runs"
```

After training the first stage, we can do model averaging using stochastic weight averaging (SWA) on the top three performing model weights to further enhance the generalization capabilities:

```
bioencoder.swa(config_path=r"bioencoder_configs/swa_stage1.yml")
```


## Learning rate finder (optional)

Using this function is entirely optional, but may used to help find appropriate learning rates for the second stage. We recommend running the LR finder several times since it is randomly intialized and may thus vary somewhat in its outcome.   

```
bioencoder.lr_finder(config_path=r"bioencoder_configs/lr_finder.yml")
```

## Training stage 2

To train stage 2 and do SWA, run the following command:

```
bioencoder.train(config_path=r"bioencoder_configs/train_stage2.yml", overwrite=True)
bioencoder.swa(config_path=r"bioencoder_configs/swa_stage2.yml")
```
