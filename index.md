---
layout: default
title: Final project
---

# We Are Never Ever Getting BERT To Get Her


## Creating the dataset

Data was scraped from https://www.azlyrics.com/lyrics/taylorswift. The scraping is provided in [this notebook](/notebooks/scraping.ipynb).

**What songs were not considered for analysis**:

* those with *Taylor's Version* but without *(From The Vault)* in the name: these are essentially duplicate songs
* those with *Remix*, *Sad Girl Autumn Version*, *Music Video Version*, *Acoutic* and the likes in the name
* select files that were rather voices memos than songs

Overall, 487 songs were available for scraping, but only 373[^1] were used in the project.

The website includes some metadata for some songs. So, the dataset looked something like this:

| Artist | Song Name |  Writers  |	Album   | Release Year  |   Lyrics  |   Pen |   Category    |   Path    |

There were NA values in the *Writers, Album, Release Year* columns because this information was not always provided on the website in a machine readable format.

There were NA values in the *Pen Category* column because this information is only available for limited songs.

I attempted to use the musicbrainz database for metadata but their API broke me and I gave up.					

## Data preparation

This is a very important (and fun!) step that is valuable for text data analysis on its own. Which lines from the songs to choose? I answer this question in [this notebook](/notebooks/data preparation.ipynb)

### N-gram vectorizer on song level

In this step, I identified identical lines in one song (~the chorus), leaving only the first occurence of this line in the song. 

This procedure was repeated for each song, but I will show it on one example. First of all, I lemmatized the song lyrics (stop words were not deleted, as they might be valuable in this step). Secondly, I used 1-gram Count vectorizer on a song. Then, I computed a similarity matrix for the lines in the song. When two lines 

**Why 1-gram?** My experiments showed that 2- and 3- gram vectorizers were, expectedly, not identifying such lines that used the same word in different order -- a poetry trick that was used often in the corpus.

**Why cosine similarity?** It was chosen at random. I do not think that the choice of the similarity function matters much in a task where we look at a high similary threshold.

Let us look at the song "Cassandra". In the figure below, yellow squares mean line combinations with the highest similarity. The figure shows 6 different sequences of yellow squares (only one part above or below the diagonal counts). These include two identical bridges, two choruses and two verse starting lines.

{{% include_relative notebooks/figures/cassandra_before.html}}


N-gram vectorizer on line level: identifying identical words and leaving only one




## Feature extraction

I am testing two approaches to my task. The first one is made possible thanks to the “Pen” classification provided by the author. I trained [an XLM[^2] model](https://huggingface.co/docs/transformers/v5.20.1/en/model_doc/xlm#transformers.XLMForSequenceClassification) on a sequence classification task and then used last hidden states of the trained model to get embeddings. The second approach is inference from a feature extraction [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-8B). 

Overall, I trained two models (both XLM, the data for the first model was full song lyrics, the data for the second were individual song lines) on a sequence classification task. Similarly, I inferenced Qwen twice: once for full song lyrics and once for individual lines.

Then, I ran four separate classification tasks. The training was done in [this](https://colab.research.google.com/drive/1Uqih-teT_H1Cm_fHVyXH5Fsbzb44HBti?usp=sharing) Google Colab notebook.

**Important notes**:

* training was done on very small batches (2) because of GPU RAM issues;
* the full lyrics model was trained for 5 epochs; validation loss grew (and overall the model was not... robust, see next section)
* the individual lines model was trained for 1 epoch only because of limited time. 1 epoch took about 20 minutes
* I encountered zero issues with Qwen inference. This is a plus if we keep in mind the major goal of this project -- to figure out what is both convenient and productive

## Quantitative comparison of the embeddings

## Qualitative comparison

## Conclusion

Overall, the limitations of my project:
* 

[^1]: yeah, Taylor Swift is impressively productive
[^2]: I was planning to use deBERTa, but I encountered a problem with training. The name of the project stayed because I loved the pun too much.