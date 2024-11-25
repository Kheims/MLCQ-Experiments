This is the repository for the experiments of the paper "Exploring NLP Techniques for Code Smell Detection: A Comparative Study." The study compares various NLP-based models for detecting code smells to a baseline model, highlighting their strengths and weaknesses.

## Prerequisites
First, create a conda environment and install the dependencies 

```
conda create -n mlcqenv python=3.10
conda activate mlcqenv
conda install -f requirements.txt
```

In order to recreate the json containing the code snippet according to the paths specified in the [MLCQ](https://zenodo.org/records/3590102) dataset 
1. First set your github token to communicate with the api, see [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) for more information on setting your token
2. Export your acquired token as an environment variable : `export GITHUB_TOKEN=<your_github_token>`
3. Run the DataExtractor script : `python DataExtractor.py`

## Baseline 
The baseline here is j48, a decision tree-based algorithm widely considered state-of-the-art for code smell detection ( see [1](https://www.mdpi.com/2076-3417/14/14/6149) and [2](https://link.springer.com/article/10.1007/s10664-015-9378-4) )

We need to first compute code metrics as they are the features needed for this model, to do so we use [Designite](https://www.designite-tools.com/), install it following the [official repo](https://github.com/tushartushar/DesigniteJava)
1. Run `python baseline/MetricsExtractor.py` to prepare the code snippets in .java files.
2. Run `python baseline/DesigniteRun.py` to execute Designite on the java files producing a DesigniteOutput file.
3. Run `baseline/DatasetCreator.py` to prepare the final dataset to feed to the model.
4. Finally, run `train.py` to train and test the model.

> **Tip:** You can speed up the Designite processing by specifying the number of workers when using `MetricsExtractor.py`. This divides the dataset into batches, enabling parallel processing for faster execution.

## Training 

There are different models each with different components, to train the final bilstm with attention model run :

```
python bilstm_attn_train.py --batch_size 16 --epochs 20 --learning_rate 0.0001  --hidden_dim 512 --num_layers 2
```

Whereas to run the CodeBert model run :

```
python bert.py
```

All the results will be stored to their corresponding log files.

## Acknowledgments

This work relies on:
- The [MLCQ dataset](https://zenodo.org/records/3590102)
- The [Designite tool](https://www.designite-tools.com/)
- CodeBert pretrained model from [huggingface](https://huggingface.co/microsoft/codebert-base)
 
