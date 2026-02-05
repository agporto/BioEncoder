# Changelog

## 1.0.5
- FIX: class order in model explorer was broken!
- FIX: interactive plot was looking for a missing argument 
- refactored code to remove torch and streamlit deprecation warnings

## 1.0.4
- FIX: allow for alpha-channel in inference

## 1.0.3
- FIX: still incongruent plotting of combined train 
- FIX: plot an augmentation sample with alpha channel imgs

## 1.0.2
- FIX: previous fixes didn't get fully merged from dev branch
- FIX: install rich

## 1.0.1
- NEW: perplexity option for interactive plots
- FIX: horizontal bar plotting in headless environments and inference without config
- FIX: batch size not multiple error, where during interactive plotting wrong bs would crash the plotting
- interactive plotting improvements (fix bootstrap/sample behaviour)
- moved and renamed configuration templates into `bioencoder_configs/`
- updates to README and citation information; improved installation help

## 1.0.0
- NEW: tested with py310 and py311
- FIX: model-explorer was broken due to internal code refactoring
- FIX: config files had faulty defaults

## 0.3.1
- FIX: export agumentations
- improved CLI inference script (better performance: model gets cached)
- better documentation in the repo and the config files

## 0.3.0
- added inference script: extract embeddings from single image
- added parameter to show samples of image augmentations, and an argument to do a training dry run
- more sensible LR finder options (e.g. to chuck more from beginning or end of range)
- better annotated config files with default values

## 0.2.1
- FIX: broken plotting function
- fixed various typos in config files
- enhanced plotting functions

## 0.2.0
- proper loggers
- cli interface
- help files

## 0.1.1

- fixed some small bugs
- readme/repo rework
- add a license

## 0.1.0

- intial release of the pypi-version
