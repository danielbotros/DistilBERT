**Dataset Motivation and Overview**
We use Wikitext for our model pretraining on masked language modeling, then IMDb, MRPC, and CoLA for evaluation. These three downstream
tasks let us evaluate our model on grammar (CoLA), semantics (MRPC), and sentiment (IMDb), giving us a well rounded set of benchmarks to evaluate language capability. Below, you may find more information on each dataset and how to use them.

**Wikitext**
The Wikitext dataset is a large collection of high-quality, cleaned English text extracted from Wikipedia articles. It contains rich, diverse, and grammatically accurate sentences that capture a wide range of topics and writing styles, making it particularly well-suited for pretraining transformer models and providing a strong foundation for downstream natural language processing tasks.

Obtained wikitext data from Hugging Face: [Wikitext](https://huggingface.co/datasets/Salesforce/wikitext)

Imported using:

```
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
```

**CoLA and MRPC**
CoLA (Corpus of Linguistic Acceptability) and MRPC (Microsoft Research Paraphrase Corpus) are part of the GLUE (General Language Understanding Evaluation) benchmark, which is widely used for evaluating language model performance on a variety of natural language understanding tasks. CoLA is a binary classification task where the goal is to determine if a given sentence is grammatically acceptable.
MRPC is a binary classification task that involves determining whether two sentences are semantically equivalent.

Obtained from Hugging Face:[GLUE](https://huggingface.co/datasets/nyu-mll/glue)

Imported using:

```
from datasets import load_dataset

# Load CoLA
cola_dataset = load_dataset("glue", "cola")
train_cola = cola_dataset["train"]
validation_cola = cola_dataset["validation"]
test_cola = cola_dataset["test"]

# Load MRPC
mrpc_dataset = load_dataset("glue", "mrpc")
train_mrpc = mrpc_dataset["train"]
validation_mrpc = mrpc_dataset["validation"]
test_mrpc = mrpc_dataset["test"]
```

** IMDb **
The IMDb dataset is a large, balanced binary sentiment classification dataset consisting of 50,000 movie reviews, with 25,000 reviews for training and 25,000 for testing. It is widely used for evaluating text classification models. The dataset includes positive and negative sentiment labels, making it ideal for sentiment analysis tasks.

Obtained from Hugging Face: [IMDb Dataset](https://huggingface.co/datasets/stanfordnlp/imdb)

Imported using:

```
from datasets import load_dataset

# Load IMDb
imdb_dataset = load_dataset("imdb")
train_imdb = imdb_dataset["train"]
test_imdb = imdb_dataset["test"]
```
