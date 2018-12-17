# CTADaily (yashl3)

CTADaily aims to provide predictions on market sentiment (up or down) for a given commodity. This is done by building a topic model for Reuters headlines from 2014 to 2016, from which we can use topic distributions as features for a regression model to determine whether or not that given commodity will go up or down. 

This GitHub repo has all headlines published from Reuters between 2007 and 2016 in a pickled file format. These are then unserialized and then converted from a file corpus to a line corpus (stored as single CSV file). These headlines are filtered for the occurence of the word 'oil'. 

The source of data is from this GitHub Repo : https://github.com/philipperemy/Reuters-full-data-set

### Technology Overview

- Pre process using NLTK Stopwords, Gensim Simple Pre Proccessor. Spacy has not been used despite it's advantages (please see Tech Review). 
- Form Corpus for a given text representation (Unigram or Bigram)
- Build LDA Mallet Model (Gensim) 
- Optimise LDA Mallet Model by Number of Topics vs Coherence Score
- Use Topic Distributions as features for Regression Model on Oil Price (Dummy Variable, see Notebook)

### Function Overview 


### Getting Setup

If you haven't already please install mallet. There is also an alternative LDA model we will allow usage of (LDA Multicore). 

https://programminghistorian.org/en/lessons/topic-modeling-and-mallet#installing-mallet

### Usage

To begin with CTADaily please navigate to the IPython Notebook, which can be run cell by cell to walkthrough topic modelling and implementation/optimisation techniques. 

-  CTADaily.ipynb	(Notebook To Run)
-  CTADaily.py	(Contains Helper Methods)
-  DCOILWTICO.csv (Data Extract from FRED)
-  WTI-LDA-TOPIC.csv	(LDA Topic Distributions)
-  WTI-NEWS.csv (Reuters Headlines - Line Corpus)

You will require a few packages to get this working : pandas, matplotlib, gensim, nltk, pyLDAvis, statsmodels, sklearn

Please run the following before beginning:
```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

If you have issues with your Mallet installation, please use the LDA Multicore model instead. 

```
lda_model = build_lda_model('Multicore', corpus, dictionary, num_topics=10)
```
