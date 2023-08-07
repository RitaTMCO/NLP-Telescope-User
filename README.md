# **NLP-Telescope**

## **Table of contents**
1) [Introduction](#introduction)
2) [Requirements for the NLP systems evaluation](#requirements)
      1) [Machine Traslation](#machine)
      2) [Dialogue System](#dialogue)
      3) [Summarization](#summarization)
      4) [Classification](#classification)

3) [Installation](#install)

4) [Web Interface](#web)

5) [Command Line Interface (CLI)](#cli)
      1) [Comparing NLG systems](#cli-nlg)
      2) [Comparing Classification systems](#cli-class)
      3) [Scoring](#cli-score)
      4) [Comparing two MT systems](#cli-compare)

---------------------------------------

## **Introduction:** <a name="introduction"></a>

NLP-Telescope is a comparative analysis tool which is an updated and extended version of MT-Telescope [(Rei et al., 2021)](https://aclanthology.org/2021.acl-demo.9/). Like MT-Telescope, it aims to facilitate researchers and developers to analyse their systems by offering features such as:

+ System-level metric that globally evaluates the outputs of the systems;
+ Segment-level metric that evaluates segment-by-segment the outputs of the systems;
+ Dynamic Corpus Filtering to filter your testset with specific linguistic phenomena such as named entities;
+ Visual interactive plots (important to compare systems side-by-side and segment-by-segment);
+ Statistical tests, in which the tool uses bootstrap resampling [(Koehn et al., 2004)](https://aclanthology.org/W04-3250/).

NLP-Telescope also offers new features compared to MT-Telescope, such as:

+ Analysing and comparing the results of N systems from M references. **N and M are numbers greater than or equal to 1.** This functionality is updated from MT-Telescope, which analyses only two systems from one reference;

+ Being able to analyse four Natural Language Processing (NLP) tasks such as **machine translation**, **text summarization**, **dialogue system** and **text classification**. For each task, **appropriate visual analysis interface, metrics and filters are added**. This functionality is updated from MT-Telescope, which analyses machine translation systems only;

+ Analysing and evaluating **gender biases** when comparing the references with the systems outputs. Only available for the machine translation;

+ Being able to **rank the systems** through an aggregation mechanism that aggregates all requested metrics;

+ Being able to **rename systems** (important for systems to contain the names that the user requires in the plot) through the upload of one file (in which each line is a system name) or directly on the web browser;

+ Being able to download plots and export tables neatly in a folder.

For Natural Language Generation (NLG) tasks such as machine translation, text summarization and dialogue system, we have three types of visual interface:

+ **Error-type analysis:** To evaluate the system utility, the tool divides the errors into four parts through the stacked bar plot. Only available if COMET or BERTScore are selected for segment-level metrics;

+ **Segment-level scores histogram:** With a histogram plot, one may observe general evaluation of the distribution of scores between systems;

+ **Pairwise comparison:** The user may choose two systems from many systems to analyse. It is used when the differences between two systems are not obvious. This type of comparison is composed by:

    + Segment-level comparison (with bubble plot, the user may check the comparison of the sentence scores of the two systems through the differences);
    + Bootstrap Resampling.

For text summarization and dialogue system, we also have the funcionally named **Similar Source Sentences**, in which the user may see the 10 sentences in the source that contain more similarity with sentences of systems outputs. The similarity is determined thought the maximum and minimum values calculated using the cosine similarity.

For the classification task, we have the following visual interfaces:

+ Confusion Matrix of a system;
+ Confusion Matrix of a system focused on one label;
+ Scores of each label for each system through the stacked bar plot;
+ Examples that are incorrectly labelled.

For all tasks and for each reference, the tool offers a table and a plot with system metrics scores. 

For each task, the tool contains a set of metrics and filters. The user may change these sets in files [metrics.yaml](user/metrics.yaml) and [filters.yaml](user/filters.yaml), located in the folder [user](user/). Same case for [bias evaluation](user/bias_evaluations.yaml) and [universal metrics](user/universal_metrics.yaml) (which the user may add more weighted-mean with different weights).

In this document, we will explain the requirements and how to install and run NLP-Telescope. To run the NLP-Telescope tool you may use:

1) the web browser;
2) the command line interface.



## **Requirements for the evaluation of NLP systems:** <a name="requirements"></a>

For the evaluation of systems, it is required three type of files: **input file**, **reference file** and **output file**. A type of file may be associated to a specific name depending on the task. 

The tool considers the file's text as a set of segments (which may be a sentence), which are organized by lines. The requirement of the number of segments per file type changes according to the task. 

In this section, we will see what are the **specific names of the file types**, their **contents**, **size requirements** and indicate **examples** for each task.


### **Machine Translation:** <a name="machine"></a>

The following files are required:

1) **Source file**: File that contains text written in one language (source language) that will be translated into another language (target language);

2) **One or more reference files**: File(s) that contain(s) text that will be the point of reference for the outputs of systems. Reference may be the human translation or the "correct" translation;

3) **One or more output files**: File(s) that contain(s) the outputs of the models which are texts translated from the source.

**All files** must have the **same number of segments**.

You may see examples of files [here](data/examples/mt/)

You must also indicate the language in which the texts are written. In the web interface, you must write the language pair of the files as, for instance, 'en-ru', in which 'en' is the source language and 'ru' is the target language. In the command line interface, you must only indicate the target language. If the language is indifferent and BERTScore metric is not used, then write X-X.

You may also upload one file in which each line is a system name. The number of lines must be equal to the number of systems and the order must be the same as the user loaded systems.


### **Dialogue System:** <a name="dialogue"></a>

The following files are required:

1) **Context file**: File that contains the context between the user and the system;

2) **One or more truth answers files**: File(s) that contain(s) the correct text that the system should have answered to;

3) **One or more systems answers files**: File(s) that contain(s) the text that the system actually answered to.

**The truth answers and the systems answers** must have the **same number of segments**.

You can see examples of files [here](data/examples/dialo/).

You must also indicate the language in which the texts are written. If the language is indifferent and BERTScore metric is not used, then write X.

You may also upload one file in which each line is a system name. The number of lines must be equal to the number of systems and the order must be the same as the user loaded systems.


### **Summarization:** <a name="summarization"></a>

The following files are required:

1) **Text to be summarized file**: File that contains the text that will se summarized by the system;

2) **One or more references files**:  File(s) that contain(s) tthe ext that will be the point of reference for the outputs of systems;

3) **One or more systems summaries files**: File(s) that contain(s) the summary produced by the system.

**The reference and the systems summaries** must have the **same number of segments**.

You can see examples of files [here](data/examples/sum/).

You must also indicate the language in which the texts are written. If the language is indifferent and BERTScore metric is not used, then write X.

You may also upload one file in which each line is a system name. The number of lines must be equal to the number of systems and the order must be the same as the user loaded systems.


### **Classification:** <a name="classification"></a>

The following files are required:

1) **Samples file**: File that contains the samples;

2) **One or more true labels files**: File(s) that contain(s) the true label of each sample;

3) **One or more predicated labels files**: File(s) that contain(s) the predicated labels of each sample.

4) **One file with all available labels**: File that contains all available labels. Each line is one label.

**Samples file, true labels files and predicated labels files** must have the **same number of segments**.

You can see examples of files [here](data/examples/class/).

You may also upload one file in which each line is a system name. The number of lines must be equal to the number of systems and the order must be the same as the user loaded systems.

## **Installation:** <a name="install"></a>

Create a virtual environment. Run: 

```bash
python3 -m venv NLP-ENV
```

Activate virtual environment. Run:

```bash
source NLP-ENV/bin/activate
```

Make sure you have [poetry](https://python-poetry.org/docs/#installation) installed, then run the following commands:

```bash
git clone https://github.com/RitaTMCO/NLP-Telescope-User.git
cd NLP-Telescope-User
git checkout evaluation
poetry install --without dev
```

Finally, run the following commands:

```bash
chmod +x download.sh 
./download.sh
```


## **Before running the tool:**

Some metrics, such as COMET, may take some time. You can switch the COMET model to a more lightweight model with the following env variable:
```bash
export COMET_MODEL=wmt21-cometinho-da
```




## **Web Interface:** <a name="web"></a>


Run the following command to display the web interface:
```bash
telescope streamlit
```
While the web browser is being displayed, it is created a folder called **downloaded_data** within folder [user](user/). The exported tables and plots downloaded from the web will be directed to the folder **downloaded_data**.




## **Command Line Interface (CLI):** <a name="cli"></a>


### **Comparing NLG systems:** <a name="cli-nlg"></a>

Run command `telescope n-compare-nlg` to compare NLG systems with CLI.

```
Usage: telescope n-compare-nlg [OPTIONS]

Options:
  -s, --source FILENAME           Source segments.  [required]
  -c, --system_output FILENAME    System candidate. This option can be
                                  multiple.  [required]
  -r, --reference FILENAME        Reference segments. This option can be
                                  multiple.  [required]
  -t, --task [machine-translation|summarization|dialogue-system]
                                  NLG to evaluate.  [required]
  -l, --language TEXT             Language of the evaluated text.  [required]
  -m, --metric [COMET|BLEU|chrF|ZeroEdit|BERTScore|TER|GLEU|ROUGE-1|ROUGE-2|ROUGE-L|Accuracy|Precision|Recall|F1-score]
                                  Metric to run. This option can be multiple.
                                  
                                  |machine-translation|: [COMET, BLEU, chrF,
                                  ZeroEdit, TER, GLEU, BERTScore].
                                  
                                  |summarization|: [ROUGE-1, ROUGE-2, ROUGE-L,
                                  BERTScore].
                                  
                                  |dialogue-system|: [BLEU, ROUGE-1, ROUGE-2,
                                  ROUGE-L, BERTScore].  [required]
  -f, --filter [named-entities|length|duplicates]
                                  Filter to run. This option can be multiple.
                                  
                                  |machine-translation|: [duplicates, length,
                                  named-entities].
                                  
                                  |summarization|: [duplicates, length, named-
                                  entities].
                                  
                                  |dialogue-system|: [duplicates, length,
                                  named-entities].
  --length_min_val FLOAT          Min interval value for length filtering.
  --length_max_val FLOAT          Max interval value for length filtering.
  --seg_metric [COMET|ZeroEdit|BERTScore|GLEU|ROUGE-L|Accuracy|F1-score]
                                  Segment-level metric to use for segment-
                                  level analysis.
  -o, --output_folder TEXT        Folder in which you wish to save plots.
  --bootstrap
  -x, --system_x FILENAME         System X NLG outputs for segment-level
                                  comparison and bootstrap resampling.
  -y, --system_y FILENAME         System Y NLG outputs for segment-level
                                  comparison and bootstrap resampling.
  --num_splits INTEGER            Number of random partitions used in
                                  Bootstrap resampling.
  --sample_ratio FLOAT            Proportion (P) of the initial sample.
  -n, --systems_names FILENAME    File that contains the names of the systems
                                  per line.
  -b, --bias_evaluations [Gender]
                                  Bias Evaluation. This option can be
                                  multiple.
                                  
                                  |machine-translation|: [Gender].
                                  
                                  |summarization|: [].
                                  
                                  |dialogue-system|: [].
  --option_gender_bias_evaluation [with dataset|with library|with datasets and library]
                                  Options for Gender Bias Evaluation.
  -u, --universal_metric [average|median|pairwise-comparison|social-choice-theory|weighted-mean-seed-12_-1_1|weighted-mean-seed-12_0_1_TER|weighted-mean-seed-24_-1_1|weighted-mean-seed-24_0_1_TER|weighted-mean-seed-36_-1_1|weighted-mean-seed-36_0_1_TER]
                                  Models Rankings from Universal Metric.
                                  
                                  |machine-translation|: [average, median,
                                  pairwise-comparison, social-choice-theory,
                                  weighted-mean-seed-12_-1_1, weighted-mean-
                                  seed-12_0_1_TER, weighted-mean-seed-24_-1_1,
                                  weighted-mean-seed-24_0_1_TER, weighted-
                                  mean-seed-36_-1_1, weighted-mean-
                                  seed-36_0_1_TER].
                                  
                                  |summarization|: [].
                                  
                                  |dialogue-system|: [].
  --help                          Show this message and exit.
```

#### Example 1: Running several metrics

Running BLEU, chrF BERTScore and COMET to compare three MT systems with two references:

```bash
telescope n-compare-nlg \
  -s path/to/src/file.txt \
  -c path/to/system-x/file.txt \
  -c path/to/system-y/file.txt \
  -c path/to/system-z/file.txt \
  -r path/to/ref-1/file.txt \
  -r path/to/ref-2/file.txt \
  -t machine-translation\
  -l en \
  -m BLEU -m chrF -m BERTScore -m COMET
```

#### Example 2: Saving a comparison report

```bash
telescope n-compare-nlg \
  -s path/to/src/file.txt \
  -c path/to/system-x/file.txt \
  -c path/to/system-y/file.txt \
  -c path/to/system-z/file.txt \
  -r path/to/ref-1/file.txt \
  -r path/to/ref-2/file.txt \
  -t machine-translation\
  -l en \
  -m BLEU -m chrF -m BERTScore -m COMET \
  --output_folder FOLDER-PATH
```

For FOLDER-PATH location, a folder is created for each reference that contains the report.



### **Comparing Classification systems:** <a name="cli-class"></a>

Run command `telescope n-compare-classification` to compare classification systems with CLI.

```
Usage: telescope n-compare-classification [OPTIONS]

Options:
  -s, --source FILENAME           Source segments.  [required]
  -c, --system_output FILENAME    System candidate. This option can be
                                  multiple.  [required]
  -r, --reference FILENAME        Reference segments. This option can be
                                  multiple.  [required]
  -l, --labels FILENAME           Existing labels  [required]
  -m, --metric [Accuracy|Precision|Recall|F1-score]
                                  Metric to run. This option can be multiple.
                                  [required]
  -f, --filter [duplicates]       Filter to run. This option can be multiple.
  --seg_metric [Accuracy|F1-score]
                                  Segment-level metric to use for segment-
                                  level analysis.
  -o, --output_folder TEXT        Folder in which you wish to save plots.
  -n, --systems_names FILENAME    File that contains the names of the systems
                                  per line.
  -u, --universal_metric []       Models Rankings from Universal Metric.
  -x, --system_x FILENAME         System X outputs for pairwise-comparison
  -y, --system_y FILENAME         System Y outputs for pairwise-comparison.
  --help                          Show this message and exit.
```

#### Example 1: Running two metrics

Running Accuracy and F1-score to compare three systems with two references:

```bash
telescope telescope n-compare-classification \
  -s path/to/input/file.txt \
  -c path/to/system-x/file.txt \
  -c path/to/system-y/file.txt \
  -c path/to/system-z/file.txt \
  -r path/to/ref-1/file.txt \
  -r path/to/ref-2/file.txt \
  -l path/to/all_labels.txt \
  -m Accuracy -m F1-score
```

#### Example 2: Saving a comparison report

```bash
telescope telescope n-compare-classification \
  -s path/to/input/file.txt \
  -c path/to/system-x/file.txt \
  -c path/to/system-y/file.txt \
  -c path/to/system-z/file.txt \
  -r path/to/ref-1/file.txt \
  -r path/to/ref-2/file.txt \
  -l path/to/all_labels.txt \
  -m Accuracy -m F1-score \
  --output_folder FOLDER-PATH
```

For FOLDER-PATH location, a folder is created for each reference that contains the report.



### **Scoring:** <a name="cli-score"></a>

To get the system level scores for a particular MT simply run `telescope score`.

```bash
telescope score -s {path/to/sources} -t {path/to/translations} -r {path/to/references} -l {target_language} -m COMET -m chrF
```



### **Comparing two MT systems:** <a name="cli-compare"></a>

For running MT system comparisons for two system with CLI, you should use the `telescope compare` command.

```
Usage: telescope compare [OPTIONS]

Options:
  -s, --source FILENAME           Source segments.  [required]
  -x, --system_x FILENAME         System X MT outputs.  [required]
  -y, --system_y FILENAME         System Y MT outputs.  [required]
  -r, --reference FILENAME        Reference segments.  [required]
  -l, --language TEXT             Language of the evaluated text.  [required]
  -m, --metric [COMET|BLEU|chrF|ZeroEdit|TER|GLEU|BERTScore]
                                  MT metric to run.  [required]
  -f, --filter [named-entities|length|duplicates]
                                  MT metric to run.
  --length_min_val FLOAT          Min interval value for length filtering.
  --length_max_val FLOAT          Max interval value for length filtering.
  --seg_metric [COMET|ZeroEdit|GLEU|BERTScore]
                                  Segment-level metric to use for segment-
                                  level analysis.
  -o, --output_folder TEXT        Folder in which you wish to save plots.
  --bootstrap
  --num_splits INTEGER            Number of random partitions used in
                                  Bootstrap resampling.
  --sample_ratio FLOAT            Proportion (P) of the initial sample.
  --help                          Show this message and exit.
```

