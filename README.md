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

NLP-Telescope is a comparative analysis tool which is an updated and extended version of MT-Telescope [(rei, et al 2021)](https://aclanthology.org/2021.acl-demo.9/). Like MT-Telescope, it aims to facilitate researchers and developers to analyze their systems by offering features such as:

+ System-level metric that globally evaluates the outputs of the systems;
+ Segment-level metric that evaluate segment-by-segment the outputs of the systems;
+ Dynamic Corpus Filtering to filter your testset with specific linguistic phenomena such as named entities;
+ Visual interactive plots. Important to compare systems side-by-side segment-by-segment.
+ Statistical tests. For this purporse, the tool use bootstrap resampling [(Koehn, et al 2004)](https://aclanthology.org/W04-3250/);

NLP-Telescope also offers new features compared to MT-Telescope such as:

+ Analyze and compare the results of N systems from M references. **N and M are numbers greater than or equal to 1.** This functionality is updated from MT-Telescope which analyzes only two systems from one reference;

+ Being able to analyze four Natural Language Processing (NLP) tasks such as: **machine translation**, **text summarization**, **dialogue system** and **text classification**. For each task, **appropriate visual analysis interface, metrics and filters are added**. This functionality is updated from MT-Telescope, which analyzes only machine translation systems;

+ Being able to **rename systems**. Important for systems to have the names that the user wants in the plot.

For Natural Language Generation (NLG) tasks such as machine translation, text summarization and dialogue system, we have three types of visual interface:

+ **Error-type analysis:** To evaluate the system utility, the tool divides the errors into four parts through the stacked bar plot. Only available if COMET or BERTScore are selected for segment-level metrics;

+ **Segment-level scores histogram:** With a histogram plot, one may observe general evaluation of the distribution of scores between systems.

+ **Pairwise comparison:** The user can choose two systems of many systems to analysis. It is used when the differences between two systems are not obvious. This type of comparison is composed by:

    + Segment-level comparison (With bubble plot, the user may check the comparison of the sentence scores of the two systems through the differences);
    + Bootstrap Resampling.

For the classification task, we have following visual interfaces:

+ Confusion Matrix of a system;
+ Confusion Matrix of a system focused on one label;
+ Scores of each label for each system through the stacked bar plot;
+ Examples that are incorrectly labelled

For all tasks and for each reference, the tool offers a table with system metrics scores. 

In this document, we will explain the requirements and how to install and run NLP-Telescope. To run the NLP-Telescope tool you can use:

1) the web browser;
2) the command line interface.





## **Requirements for the NLP systems evaluation:** <a name="requirements"></a>

For the evaluation of systems, it is required three type of files: **input file**, **reference file** and **output file**. A type of file may have a specific name depending on the task. 

The tool considers the file's text as a set of segments (which can be a sentence) and are organized by lines. The requirement of the number of segments per file type changes according to the task. 

In this section, we will see what are the **specific names of the file types**, their **contents**, **size requirements** and indicate **examples** for each task.


### **Machine Translation:** <a name="machine"></a>

The following files are required:

1) **Source file**: File that contains text written in one language (source language) that will be translated into another language (target language);

2) **One or more refrerence files**: File(s) that contains text that will be the point of reference for the outputs of systems. Reference can be the human translation or the "correct" translation;

3) **One or more output files**: File(s) that contains the outputs of the models which are texts translated from the source.

**All files** must have the **same number of segments**.

You can see examples of files [here](data/mt/)

You must also indicate the language in which the texts are written. In the web interface, you must write the language pair of the files as follows 'en-ru', in which en is the source language and ru is the target language. In the command line interface, you must only indicate the target language. If the language is indifferent and BERTScore metric is not used, then write X-X.


### **Dialogue System:** <a name="dialogue"></a>

The following files are required:

1) **Context file**: File that contains the context between the user and the system;

2) **One or more truth answers files**: File(s) that contain(s) the correct text that the system should have answered to;

3) **One or more systems answers files**: File(s) that contain(s) the text that the system actually answered to.

**The truth answers and the systems answers** must have the **same number of segments**.

You can see examples of files [here](data/dialo/)

You must also indicate language that texts are. If the language is indifferent and BERTScore metric is not used, then write X.



### **Summarization:** <a name="summarization"></a>

The following files are required:

1) **Text to be summarized file** (input file): File that contains text that will se summarized by the system;

2) **One or more references files**:  File(s) that contain(s) text that will be the point of reference for the outputs of systems;

3) **One or more systems summaries files**: File(s) that contain(s) the summary produced by system.

**The reference and the systems summaries** must have the **same number of segments**.

You can see examples of files [here](data/sum/)

You must also indicate the language in which texts are written. If the language is indifferent and BERTScore metric is not used, then write X.



### **Classification:** <a name="classification"></a>

The following files are required:

1) **Samples file** (input file): File that contains the samples;

2) **One or more true labels files**: File(s) that contain(s) the true label of each sample;

3) **One or more predicated labels files**: File(s) that contain(s) the predicated labels of each sample.

**All files** must have the **same number of segments**.

You can see examples of files [here](data/class/)

You must also indicate labels. 


## **Installation:** <a name="install"></a>

Create a virtual environment. Run 

```bash
python3 -m venv NLP-ENV
```

Activate virtual environment. Run:

```bash
source NLP-ENV/bin/activate
```

Make sure you have [poetry](https://python-poetry.org/docs/#installation) installed, then run the following comands:

```bash
git clone https://github.com/RitaTMCO/NLP-Telescope
cd NLP-Telescope
poetry install --without dev
```

## **Before running the tool:**

Some metrics like COMET can take some time. You can switch the COMET model to a more lightweight model with the following env variable:
```bash
export COMET_MODEL=wmt21-cometinho-da
```




## **Web Interface:** <a name="web"></a>


Run the following command to display the web interface:
```bash
telescope streamlit
```




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
  -t, --task [machine-translation|dialogue-system|summarization]
                                  NLG to evaluate.  [required]
  -l, --language TEXT             Language of the evaluated text.  [required]
  -m, --metric [COMET|BLEU|chrF|ZeroEdit|BERTScore|TER|GLEU|ROUGE-1|ROUGE-2|ROUGE-L|Accuracy|Precision|Recall|F1-score]
                                  Metric to run. This option can be multiple.
                                  [required]
  -f, --filter [named-entities|length|duplicates]
                                  Filter to run. This option can be multiple.
  --length_min_val FLOAT          Min interval value for length filtering.
  --length_max_val FLOAT          Max interval value for length filtering.
  --seg_metric [COMET|ZeroEdit|BERTScore|GLEU|ROUGE-L|Accuracy]
                                  Segment-level metric to use for segment-
                                  level analysis.
  -o, --output_folder TEXT        Folder you wish to use to save plots.
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
  -l, --label TEXT                Existing labels  [required]
  -m, --metric [Accuracy|Precision|Recall|F1-score]
                                  Metric to run. This option can be multiple.
                                  [required]
  -f, --filter [duplicates]       Filter to run. This option can be multiple.
  --seg_metric [Accuracy]         Segment-level metric to use for segment-
                                  level analysis.
  -o, --output_folder TEXT        Folder you wish to use to save plots.
  -n, --systems_names FILENAME    File that contains the names of the systems
                                  per line.
  --help                          Show this message and exit.
```

#### Example 1: Running two metrics

Running Accuracy and F1-score to compare three systems with two references:

```bash
telescope telescope n-compare-classification \
  -s path/to/src/file.txt \
  -c path/to/system-x/file.txt \
  -c path/to/system-y/file.txt \
  -c path/to/system-z/file.txt \
  -r path/to/ref-1/file.txt \
  -r path/to/ref-2/file.txt \
  -l label-1 \
  -l label-2 \
  -l label-3 \
  -m Accuracy -m F1-score
```

#### Example 2: Saving a comparison report

```bash
telescope telescope n-compare-classification \
  -s path/to/src/file.txt \
  -c path/to/system-x/file.txt \
  -c path/to/system-y/file.txt \
  -c path/to/system-z/file.txt \
  -r path/to/ref-1/file.txt \
  -r path/to/ref-2/file.txt \
  -l label-1 \
  -l label-2 \
  -l label-3 \
  -m Accuracy -m F1-score
  --output_folder FOLDER-PATH
```

For FOLDER-PATH location, a folder is created for each reference that contains the report



### **Scoring:** <a name="cli-score"></a>

To get the system level scores for a particular MT simply run `telescope score`.

```bash
telescope score -s {path/to/sources} -t {path/to/translations} -r {path/to/references} -l {target_language} -m COMET -m chrF
```



### **Comparing two MT systems:** <a name="cli-compare"></a>

For running MT system comparisons for two system with one  with CLI you should use the `telescope compare` command.

```
Usage: telescope compare [OPTIONS]

Options:
  -s, --source FILENAME           Source segments.  [required]
  -x, --system_x FILENAME         System X MT outputs.  [required]
  -y, --system_y FILENAME         System Y MT outputs.  [required]
  -r, --reference FILENAME        Reference segments.  [required]
  -l, --language TEXT             Language of the evaluated text.  [required]
  -m, --metric [COMET|BLEU|chrF|TER|GLEU|ZeroEdit|BERTScore]
                                  MT metric to run.  [required]
  -f, --filter [named-entities|length|duplicates]
                                  MT metric to run.
  --length_min_val FLOAT          Min interval value for length filtering.
  --length_max_val FLOAT          Max interval value for length filtering.
  --seg_metric [COMET|GLEU|ZeroEdit|BERTScore]
                                  Segment-level metric to use for segment-
                                  level analysis.
  -o, --output_folder TEXT        Folder you wish to use to save plots.
  --bootstrap
  --num_splits INTEGER            Number of random partitions used in
                                  Bootstrap resampling.
  --sample_ratio FLOAT            Folder you wish to use to save plots.
  --help                          Show this message and exit.
```

