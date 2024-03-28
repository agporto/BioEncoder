
# Interactive plots

This repo is supplied with [interactive](https://bokeh.org/) PCA and T-SNE visualizations so that you can check the embeddings you get after the training. To generate the interactive plot, use:

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

