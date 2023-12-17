# Knowledge-Augmented Language Model Verification

Official Code Repository for the paper - Knowledge-Augmented Language Model Verification (EMNLP 2023): https://arxiv.org/abs/2310.12836.

## Requirements

* Python 3.8.16
* PyTorch 2.0.0
* transformers 4.28.1

## Preprocessing

Run the command below, in order to preprocess the datasets for open-domain question answering (e.g., NaturalQuestions and HotpotQA).
```sh
$ python ./preprocess/process_odqa.py
```

Note that, for the knowledge-graph question answering datasets (e.g., WebQSP and Mintaka), please download them using the following link: https://1drv.ms/f/s!Aj5JerV8SMDyi-tJ5QOlll_ksf1fRA?e=6HOmI2, and then put them in the dataset folder (e.g., ./datasets/webq for WebQSP or ./datasets/mintaka for Mintaka). 

## Run

The following command line runs the experiments for our KALMV on both NaturalQuestions and HotpotQA datasets. The experiments were conducted on a GPU with at least 24 GB of memory
```sh
$ sh ./scripts/odqa_run.sh
```

The following command line runs the experiments for our KALMV on both WebQuestions and Mintaka datasets. The experiments were conducted on a GPU with at least 24 GB of memory
```sh
$ sh ./scripts/kgqa_run.sh
```

## Evaluation

The following command line evalulates the **runned experiments** (by the above command) for our KALMV on both NaturalQuestions and HotpotQA datasets.
```sh
$ sh ./scripts/odqa_eval.sh
```

The following command line evalulates the **runned experiments** (by the above command) for our KALMV on both WebQuestions and Mintaka datasets.
```sh
$ sh ./scripts/kgqa_eval.sh
```
