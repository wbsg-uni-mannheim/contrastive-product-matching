# Contrastive Product Matching

This repository contains the code and data download links to reproduce the experiments of the paper "Supervised Contrastive Learning for Product Matching" by Ralph Peeters and Christian Bizer. [ArXiv link](https://arxiv.org/abs/2202.02098). A comparison of the results to other systems using different benchmark datasets is found at [Papers with Code - Entity Resolution](https://paperswithcode.com/task/entity-resolution/).

* **Requirements**

    [Anaconda3](https://www.anaconda.com/products/individual)

    Please keep in mind that the code is not optimized for portable or even non-workstation devices. Some of the scripts may require large amounts of RAM (64GB+) and GPUs. It is advised to use a powerful workstation or server when experimenting with some of the larger files.

    The code has only been used and tested on Linux (CentOS) servers.

* **Building the conda environment**

    To build the exact conda environment used for the experiments, navigate to the project root folder where the file *contrastive-product-matching.yml* is located and run ```conda env create -f contrastive-product-matching.yml```
    
    Furthermore you need to install the project as a package. To do this, activate the environment with ```conda activate contrastive-product-matching```, navigate to the root folder of the project, and run ```pip install -e .```

* **Downloading the raw data files**

    Navigate to the *src/data/* folder and run ```python download_datasets.py``` to automatically download the files into the correct locations.
    You can find the data at *data/raw/*

    If you are only interested in the separate datasets, you can download the [WDC LSPC datasets](http://webdatacommons.org/largescaleproductcorpus/v2/index.html#toc6) and the [deepmatcher splits](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md) for the abt-buy and amazon-google datasets on the respective websites. 
	
* **Processing the data**

    To prepare the data for the experiments, run the following scripts in that order. Make sure to navigate to the respective folders first.
    
    1. *src/processing/preprocess/preprocess_corpus.py*
    2. *src/processing/preprocess/preprocess_ts_gs.py*
    3. *src/processing/preprocess/preprocess_deepmatcher_datasets.py*
    4. *src/processing/contrastive/prepare_data.py*
	5. *src/processing/contrastive/prepare_data_deepmatcher.py*

* **Running the Contrastive Pre-training and Cross-entropy Fine-tuning**

    Navigate to *src/contrastive/*
    
	You can find respective scripts for running the experiments of the paper in the subfolders *lspc/* *abtbuy/* and *amazongoogle/*. Note that you need to adjust the file path in these scripts for your system (replace ```your_path``` with ```path/to/repo```).
	
	* **Contrastive Pre-training**
	
		To run contrastive pre-training for the abtbuy dataset for example use 

		```bash abtbuy/run_pretraining_clean_roberta.sh BATCH_SIZE LEARNING_RATE TEMPERATURE (AUG)```

		You need to specify batch site, learning rate and temperature as arguments here. Optionally you can also apply data augmentation by passing an augmentation method as last argument (use ```all-``` for the augmentation used in the paper).

		For the WDC Computers data you need to also supply the size of the training set, e.g. 

		```bash lspc/run_pretraining_roberta.sh BATCH_SIZE LEARNING_RATE TEMPERATURE TRAIN_SIZE (AUG)```
	
	* **Cross-entropy Fine-tuning**
	
		Finally, to use the pre-trained models for fine-tuning, run any of the fine-tuning scripts in the respective folders, e.g. 

		```bash abtbuy/run_finetune_siamese_frozen_roberta.sh BATCH_SIZE LEARNING_RATE TEMPERATURE (AUG)``` 

		Please note, that BATCH_SIZE refers to the batch size used in pre-training. The fine-tuning batch size is locked to 64 but can be adjusted in the bash scripts if needed.

		Analogously for fine-tuning WDC computers, add the train size: 

		```bash lspc/run_finetune_siamese_frozen_roberta.sh BATCH_SIZE LEARNING_RATE TEMPERATURE TRAIN_SIZE (AUG)```

	
--------

Project based on the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science/). #cookiecutterdatascience
