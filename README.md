# DistilBERT

## Introduction

The paper “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter” by Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf, proposed using knowledge distillation during pretraining to compress BERT models.

Our project replicates this by training and fine-tuning DistilBERT on downstream tasks like sentiment analysis, grammar judgment, and paraphrase detection, retaining 94.35% of BERT's performance across tasks.

## Chosen Result

We aimed to reproduce the results from Table 1 and Table 2 of the paper (shown below), which demonstrate that DistilBERT retains most of BERT's performance on downstream tasks--namely, classification and GLUE Benchmark tasks. We focused evaluating the model's performance on CoLA, MRPC, and IMDb tasks.

![Comparison](./results/orig_paper_tables/table1.png)

![Comparison](./results/orig_paper_tables/table2.png)

![Comparison](./results/results_comparison.png)

## GitHub Contents

**code/** contains our final training notebook and a folder containing notebooks we used to run preliminary experiments on before training our final models. Within the experiments folder, we experimented with pretraining a DistilBERT model and evaluating it on a downstream task (cola_pretrain_eval.ipynb and imdb_pretrain_eval.ipynb) as well as doing a simple task-specific distillation (mrpc_task_distillation_eval.ipynb and imdb_task_distillation_eval.ipynb).

**data/** contains a README.md citing the sources of, explaining, and showing how to use our datasets

**poster/** contains the PDF file of our poster

**report/** contains the PDF file of our final report, which entails more details about the project

**results/** contains the figures and tables generated by our code. The top level figures and models represent the results from our final training. Each downstream task folder (i.e. imdb_final_finetune has it's own set of figures for evaluation). Experimental results and models can be found under experiment_results/ organized by the experiment corresponding to its code/ notebook. Some results may not apear in this folder e.g. total speed up, but are a result of aggregation cells that can be found in the notebook.

## Re-implementation Details

To reproduce the results of the paper, we followed the knowledge distillation approach from the paper. We used the pretrained uncased BERT as the teacher model and DistilBERT as the student model. We reinitialized the student model layers by taking every other layer from the teacher model and removing the token-type embeddings and poolers.

The models were trained for 10 epochs on the wikitext dataset--a much smaller dataset than the dataset that the original paper distilled on--due to time and compute constraints. (The original paper distilled the student BERT on a concatenation of Wikipedia and Toronto Book Corpus for a duration of 90 hours). Ours took around 3 hours roughly. We trained the student model using a masked language modeling (MLM) task using the triple loss objective described in the paper: a linear combination of MLM Loss, KL-Divergence Loss, and Cosine Embedding Loss (teacher's weights are frozen during this distillation). We ran the model on Colab using the free T4 GPUs. For evaluating the student's performance, we evaluated both the student and teacher on the same downstream tasks. In particular, we fine-tuned both the student and the teacher BERT using the same finetune settings/hyperparameters on the downstream tasks such as CoLA, MRPC, and IMDb.

Challenges included the limited computational resources available on Colab but the approach followed the original paper's methodology, ensuring consistency in the distillation process.

## Reproduction Steps

To run our code, download the Jupyter Notebook from /code and import it to Google Colab. Before running the notebook, switch the Runtime Type to T4 GPU.

**!! NOTE: Most notebooks rely on hardcoded save paths that correspond to our file structure to save models and generate figures. You will likely have to edit these so that folder names / directories will be compatible with your environment !!**

## Results/Insights

We found that our implementation produced results that are very similar to the original paper's results. The plots (generated by the Colab notebook) show that our DistilBERT model and the paper's DistilBERT model perform pretty similarly on all of the downstream tasks.

Our DistilBERT retained 94.35% of the BERT model's performance, while the paper's DistilBERT retained 97% of BERT's performance.

## Conclusion

Knowledge-distilled models are more accessible, environmentally sustainable, resource efficient, and equally as effective as large models. Thus, they serve as useful alternatives to large language models in real-world, low-latency, resource-constrained applications. This project was a valuable learning opportunity to delve into a cutting-edge research topic (both theoretically and through actual implementation) that directly relates to what we learned in class. It also taught us to learn to reimplement the original paper's results using the limited resources we had.

## References

Papers:

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv. https://arxiv.org/abs/1910.01108
- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., & Bowman, S. R. (2019). GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. arXiv. https://arxiv.org/abs/1804.07461

Tools and Frameworks:

- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2020). Transformers: State-of-the-art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 38–45. https://doi.org/10.18653/v1/2020.emnlp-demos.6
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, 8026–8037. https://papers.nips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

Datasets:

- Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv. https://arxiv.org/abs/1609.07843 (for WikiText-2)
- Socher, R., Perelygin, A., Wu, J. Y., Chuang, J., Manning, C. D., Ng, A. Y., & Potts, C. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, 1631–1642. (for IMDb)
- Dolan, W. B., & Brockett, C. (2005). Automatically constructing a corpus of sentential paraphrases. Proceedings of the Third International Workshop on Paraphrasing (IWP2005). (for MRPC)
- Warstadt, A., Singh, A., & Bowman, S. R. (2019). Neural Network Acceptability Judgments. Transactions of the Association for Computational Linguistics, 7, 625–641. https://aclanthology.org/Q19-1042/ (for CoLA)

## Acknowledgments

We would like to thank CS 4782 for providing the resources that enabled us to complete this project. We also acknowledge the collaborative efforts of our team members: Daniel Botros, Jenny Chen, Jessica Cho, Eric Gao, and Xinyu Ge, whose contributions were essential in completing this project.
