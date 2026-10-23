---
layout: default
title: Final project
---


# We Are Never Ever Getting BERT To Get Her

* Machine learning helps meaningfully group large amounts of text into smaller subgroups based on researcher’s chosen parameters.

* When the researcher has particular division parameters in mind, which type of models works better for text classification or clusterization? The answer to this question may determine the methodology of the researcher’s work. 

**I plan to contribute to the answer of this question with this project.**

I am comparing the performance of two models – [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-8B) and [XLM[^2]](https://huggingface.co/docs/transformers/v5.20.1/en/model_doc/xlm#transformers.XLMForSequenceClassification). The former is specifically designed for feature extraction, the latter is to be used for downstream tasks like classification. 

I am testing the models on the dataset featuring Taylor Swift’s lyrics.
* limited data is frequent in literary analysis. 
* there is an available classification – provided by the author herself – of a set of the songs that may be used as an imitation of the researcher prior knowledge. 

Taylor Swift classifies her songs into three categories: **Quill Lyrics, Fountain Pen Lyrics, and Glitter Gel Pen Lyrics**[^3]. She has curated three respective playlists with some of her songs that would fit into each category. The majority of the songs from her catalogue were not featured in these playlists. 

**Essentially, in this project I will conduct a case study that sheds lights on what (formally) makes a song fit into each category and how other Taylor Swift songs would be classified using these labels.**

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

This procedure was repeated for each song, but I will show it on one example. First of all, I lemmatized the song lyrics (stop words were not deleted, as they might be valuable in this step). Secondly, I used 1-gram Count vectorizer on a song. Then, I computed a similarity matrix for the lines in the song. When two or more lines have cosine similarity > 0.8, I delete all of them expect for the first one.

**Why 1-gram?** My experiments showed that 2- and 3- gram vectorizers were, expectedly, not identifying such lines that used the same word in different order -- a poetry trick that was used often in the corpus.

**Why cosine similarity?** It was chosen at random. I do not think that the choice of the similarity function matters much in a task where we look at a high similary threshold.

Let us look at the song "Cassandra". In the figure below, yellow squares mean line combinations with the highest similarity. The figure shows 6 different sequences of yellow squares (only one part above or below the diagonal counts). These include two identical bridges, two choruses and two verse starting lines.

[Figure before filtering](cassandra0.html)


[Figure after filtering](cassandra1.html)

<iframe 
  src="_includes/cassandra1.html" 
  width="100%" 
  height="500px" 
  style="border:none;">
</iframe>

<iframe 
  src="cassandra1.html" 
  width="100%" 
  height="500px" 
  style="border:none;">
</iframe>

{% include cassandra0.html %}

However, this operation was not enough for some of the songs. See this [funny file](misc/similar_lines.txt), where I exported some similar lines throuout the corpus. What helped was adding additional filters:
```
if len(unique_words) < 2:
            lines_delete = np.append(lines_delete, line_i)
        if np.array(len_unique_words).mean() <= 2.5:
            lines_delete = np.append(lines_delete, line_i)
        if np.array(len_unique_words).mean() <= 4:
            if 'yeah' in unique_words or 'ooh' in unique_words:
                lines_delete = np.append(lines_delete, line_i)
        if loc[line_i] in ['Taylor Swift', 'Ed Sheeran', 'Justin Vernon']:
            lines_delete = np.append(lines_delete, line_i)
```
Overall, there were 11009 individual lyrics lines.

## Feature extraction

I am testing two approaches to my task. The first one is made possible thanks to the “Pen” classification provided by the author. I trained [an XLM model](https://huggingface.co/docs/transformers/v5.20.1/en/model_doc/xlm#transformers.XLMForSequenceClassification) on a sequence classification task and then used last hidden states of the trained model to get embeddings. The second approach is inference from a feature extraction [Qwen](https://huggingface.co/Qwen/Qwen3-Embedding-8B). 

Overall, I trained two models (both XLM, the data for the first model was full song lyrics, the data for the second were individual song lines) on a sequence classification task. Similarly, I inferenced Qwen twice: once for full song lyrics and once for individual lines.

Then, I ran four separate classification tasks. The training was done in [this](https://colab.research.google.com/drive/1Uqih-teT_H1Cm_fHVyXH5Fsbzb44HBti?usp=sharing) Google Colab notebook.

**Important notes**:

* training was done on very small batches (2) because of GPU RAM issues;
* the full lyrics model was trained for 5 epochs; validation loss grew (and overall the model was not... robust, see next section)
* the individual lines model was trained for 1 epoch only because of limited time. 1 epoch took about 20 minutes
* I encountered zero issues with Qwen inference. This is a plus if we keep in mind the major goal of this project -- to figure out what is both convenient and productive
* I could only make XML embeddings for 1000 random lines, because the file was already too large

Links to the embeddings:

[Qwen full songs](embeddings/qwen_full_song_embeddings.pkl)

[Qwen individual lines](embeddings/qwen_list_lyrics_embeddings.pkl)

[XML full songs]()

[XML 1000 random lines](https://drive.google.com/file/d/1jmIvOdsFVLJjWU-AD-h_SjcdVujuUE2x/view?usp=sharing)

## Quantitative comparison of the embeddings

With Pen Category labels (55 songs, 1843 lines), I ran four different classification tasks. The size of the song dataset (55 songs...) is laughable and, expectedly, does not show good results. Line classification seemed promising.

### Qwen embeddings

I used Qwen embeddings and support vector machines to perform classification tasks.

Results for the song model:
```
                 precision    recall  f1-score   support

      Quill Pen       0.00      0.00      0.00         3
   Fountain Pen       0.45      1.00      0.62         5
Glitter Gel Pen       0.00      0.00      0.00         3

       accuracy                           0.45        11
      macro avg       0.15      0.33      0.21        11
   weighted avg       0.21      0.45      0.28        11

Confusion Matrix:

array([[0, 3, 0],
       [0, 5, 0],
       [0, 3, 0]])
```

Results for the lines model:
```
                 precision    recall  f1-score   support

      Quill Pen       0.49      0.15      0.22       117
   Fountain Pen       0.45      0.91      0.60       159
Glitter Gel Pen       0.91      0.11      0.19        93

       accuracy                           0.47       369
      macro avg       0.61      0.39      0.34       369
   weighted avg       0.58      0.47      0.38       369

Confusion Matrix:

array([[ 17, 100,   0],
       [ 13, 145,   1],
       [  5,  78,  10]])
```

### XML model

The model had a classification head for 3 labels.

Results for the song model:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         4
           1       1.00      1.00      1.00         2

    accuracy                           1.00         6
   macro avg       1.00      1.00      1.00         6
weighted avg       1.00      1.00      1.00         6

array([[4, 0],
       [0, 2]])
```

Results for the lines model:
```
                 precision    recall  f1-score   support

      Quill Pen       1.00      1.00      1.00        55
   Fountain Pen       1.00      1.00      1.00        93
Glitter Gel Pen       1.00      1.00      1.00        37

       accuracy                           1.00       185
      macro avg       1.00      1.00      1.00       185
   weighted avg       1.00      1.00      1.00       185

array([[55,  0,  0],
       [ 0, 93,  0],
       [ 0,  0, 37]])
```

The results for the XML model are quite suspicious...

## Qualitative comparison 

Separate page

## Conclusion

I can make the following conclusions:

Overall, the limitations of my project:

* using either full songs or individual song lines. As experience showed, neither of the choices is good. Knowing this, I would use lyric blocks (individual choruses, bridges, verses) instead.

[^1]: yeah, Taylor Swift is impressively productive
[^2]: I was planning to use deBERTa, but I encountered a problem with training. The name of the project stayed because I loved the pun too much.
[^3]: https://www.hollywoodreporter.com/music/music-news/taylor-swift-songwriting-process-nashville-speech-1235224700