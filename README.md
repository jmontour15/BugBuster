# BugBuster
Jackson Montour - 2025
Antibiotic resistance classification of bacterial amino acid sequence.
### Table of Contents
1. [Project Background and Motivation](#background-and-motivation)
2. [Methods](#methods)
3. [Results](#results)
4. [Discussion](#discusion)
5. [Usage Guide](#usage-guide)

### Background and Motivation
BugBuster is an end-to-end antibiotic resistance prediction tool for bacterial protein sequences. Deep Learning models have shown immense success in genomic and proteomic applications over the past couple of years, opening the proverbial floodgates to mass usage of these powerful tools across the world of bioinformatics. Biological deep learning models were initially developed and used as single-task tools, whether that be AlphaFold for protein structure prediction, ProteinMPNN for residue prediction, RFdiffusion for de novo protein design, and countless others. These models were, and still are, state-of-the-art performers for each of their specific tasks, with AlphaFold and RFdiffusion sharing [The Nobel Prize in Chemistry 2024](https://www.nobelprize.org/prizes/chemistry/2024/press-release/).

While these developments were happening in the bioinformatics space, machine learning researchers interesting in natural language processing (NLP) were making remarkable discoveries of their own, with a specific example being the development of the BERT architecture ([Devlin et al. 2018](https://arxiv.org/pdf/1810.04805)). BERT stands for Bidirectional Encoder Representations from Transformers, with the key word being "bidirectional". BERT aimed to combat the weakness of other language models of the time — they could only look at tokens up until the current token. This means they could only read and process in one direction. This is problematic, as the context behind a certain word or phrase in a large paragraph could mean very different things depending on what follows. BERT was designed to remove this issue through an innovative pre-training method, as described in their publication:
> "The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer."

The results of the paper speak for themselves, with a particularly interesting part being the model's high performance in transfer learning and fine-tuning tasks. These results ended up inspiring the creation of Meta Research's Evolutionary Scale Modeling (ESM) protein language model ([Lin et al. 2022](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf)). This model employed BERT's strategy of MLM pre-training on amino acid sequences, and was proposed as a tool that not only achieved SOTA performance in protein folding tasks, but could also be used in a variety of transfer learning applications. This transfer learning success is a key part of ESM's usefulness, further proved by BugBuster, which uses ESM as a fine-tuning base to classify bacterial amino acid sequences as conferring antibiotic resistance or susceptibility.

As bacteria evolves and encounters various antibiotics, resistance will inevitably develop, and this is ever true given the widespread use of antibiotics as a treatment for bacterial infections. There is even conclusive evidence showing that some antibiotic resistance genes (ARGs) predate clinical antibiotic use by about 30,000 years ([D'Costa et al. 2011](https://www.nature.com/articles/nature10388)). This inevitability of ARGs makes BugBuster especially relevant as the prevalence of ARGs increases. A big part of handling antibiotic resistance in practice is profiling antibiotic resistance, which is done to specify treatment to each patient and each infection. The downside of this profiling is that the current process can take multiple days to recieve results, leaving room for the disease state of a patient to rapidly deteriorate. BugBuster is able to rapidly classify bacterial amino acid sequences as conferring antibiotic resistance or susceptibilty, which could rapidly reduce treatment time and provide healthcare practitioners with important information used to reduce the spread of ARGs.

### Methods
One of the main things holding back deep learning in genomics and proteomics is the lack of task-specific datasets. This lack of clean and professionally annotated data is only amplified when considering the most robust proteomics deep learning architecture, the transformer, is especially data hungry when compared to other popular architecures. The dataset used is the Antibiotic Resistance Statistics (ARSS-90) Dataset, published by [Dong et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11241484/#_ad93_). The transformer architecure not only performs exceptionally well (beating Dong et al. published accuracy by 7%), but also allows for exceptional explainability in the form of attention maps. The BugBuster approach leverages transfer learning to mediate the data problem, while retaining all of the benefits of the transformer architecture for protein modeling. 

BugBuster takes inspiration from the ESM-2 publication, where they use a protein language model trained with an MLM objective as a fine-tuning base combined with a structure prediction trunk to achieve SOTA results (Figure 1, top). BugBuster mimics this by using the same pretrained ESM model but with the addition of antibiotic resistance protein prediction layers (Figure 1, bottom). The modularity of this approach is utilized in BugBuster's design, as it would not require many code adjustments to train BugBuster on an entirely different task, including regression tasks.

<p align="center">
  <img src="./Figures/ESM Schematic.jpg" alt="ESM schematic" width="300px" />
  <br>
  <img src="./Figures/BugBuster Schematic.jpg" alt="BugBuster Schematic" width="300px" />
</p>

<i align="center">Figure 1 - <b>Top: </b>Example schematic from ESM publication showing the use of the model for a structure prediction task. <b>Bottom: </b>ESM schematic adapted to show how BugBuster adapts a similar approach, replacing structure prediction layers with ARG prediction layers.</i>

To experiment further, three methods of transfer learning were tested through the BugBuster pipeline:

1. Unfrozen ESM weights
2. Frozen ESM weights
3. Low-Rank Adaptation (LoRA)

### Results
Overall, the BugBuster model performs very impressively with all three weight freezing methods. The best performing model was trained with unfrozen ESM weights and reaches an overall accuracy of 95% across both classes, which surpasses the best published accuracy values by 7% (Figure 2). The best performing model previously published is trained from scratch on the ARSS-90 dataset.

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="./Figures/Performance Comparison.jpg" alt="Performance Comparison" style="margin-top: 10px">
</div>
<i style="text-align: center; display: block; width: 100%; max-width: 600px; margin: 0 auto;">
    Figure 1 - <b>A: </b>Performance of fully trained BugBuster with unfrozen ESM weights throughout training. 
    <b>B: </b>Performance of other popular approaches for protein classification used on ARSS-90 dataset, 
    taken from <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC11241484/#_ad93_" target="_blank">Dong et al. 2024</a>.
</i>
<br>
Also implemented through BugBuster are two attention visualization techniques: attention weight maps, motif plots. Examples of these two types of figures can be seen in Figure 3. These plots are especially useful as they offer very simple ways to visualize the decision making of the model. These methods, however, are not perfect and their usefulness can depend on model architecture and training objective.

<div style="display: flex; flex-direction: column; align-items: center; margin: 20px 0;">
    <img src="./Figures/Example Attention Maps.jpg" alt="Attention Matrices" style="height: 150px;">
    <img src="./Figures/Example Motif Plots.jpg" alt="Motif Plots" style="height: 200px; margin-bottom: 10px;">
    <i style="text-align: center; width: 100%; max-width: 600px; margin: 0 auto;">Figure 3 - <b>Top: </b>Example attention matrix plots generated from bacterial ribonuclease sequence, correctly predicted as not antibiotic resistant. <b>Bottom: </b>Example motif plots generated from two multidrug resistant proteins, correctly identified by BugBuster.</i>
</div>

### Discussion
If BugBuster shows anything, its that ESM can be used to create robust transfer learning pipelines, further adding to the impressive resume of the BERT and ESM architectures. Unfortunately, Figure 3's attention matrices and motif plots appear to be quite noisy as it relates to ARG classification. This is most likely due to the varrying training objectives and overall task complexity. The attention matrix in the very top-left was collected from only ESM layers, which is why there is little noise. 

ESM's MLM training objective leads the model to understand relative and position specific importances throughout the sequence, which is why we see a very clean self-attention line that spans from the upper-left to lower-right ends of the plot. This is almost entirely thrown out the window when moving one plot to the right. The classification head layers are learning specifically to classify proteins based on the lower-level features recognized by the initial ESM layers. The classification layers then put these low-level features together to form very high-level abstractions, making them difficult to visualize. 

The motif plots follow this same logic as they were both collected from the classification head layers. In an attempt to increase interpretability, the weights were even cube scaled to try and increase differences between high and low weight tokens. There are other more robust interpretability techniques for protein models, but they were considered outside of the scope of this project.

### Usage Guide
Usage guide can be found [here](Usage.ipynb)
