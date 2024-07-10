
# Interactive plots

BioEncoder features [interactive](https://bokeh.org/) PCA and T-SNE visualizations so that you can check the embeddings you get after the training. To generate the interactive plot, use:

```python
bioencoder.interactive_plots(config_path=r"bioencoder_configs/plot_stage1.yml")
```

Interactive plots of the first stage allow the exploration of the embedding space and distances between individuals.

<p align="center"><img src="https://github.com/agporto/BioEncoder/raw/main/assets/bioencoder-interactive-plot.gif" width="600"></p>

# Model explorer

We also provide a model visualization playground, where individuals can get further insight into their data. To launch the app and explore the final classification model, simply use:

```python
bioencoder.model_explorer(config_path=r"bioencoder_configs/explore_stage1.yml")
bioencoder.model_explorer(config_path=r"bioencoder_configs/explore_stage2.yml")
```

Model visualization techniques vary between `first` and `second` stage, so make sure to select the appropriate ones.

# Inference

To extract embeddings (stage 1) or to classify (stage 2) individual images using a trained model, supply a config file (to specify the stage and the image size at training) and a path to an image (or a numpy array) to the inference script:

```python
bioencoder.inference(config_path=r"bioencoder_configs/inference.yml", image="path/to/image.jpg")
```

Note that in CLI mode you can either supply a path to an single image, or to a directory of images. There a few more options - check them with help:

```python
bioencoder_inference --config-path "bioencoder_configs/inference.yml" --path "path/to/image"

optional arguments:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH		Path to the YAML configuration file to create interactive plots.
  --path PATH          			Path to image or folder to embedd / classify.
  --save-path SAVE_PATH 		Path to CSV file with results.
  --overwrite           		Overwrite CSV file with results.
```

