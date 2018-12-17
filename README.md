# CTADaily

CTADaily aims to provide predictions on market sentiment for a given commodity. This is done by building a topic model for Reuters headlines from 2014 to 2016, from which we can use topic distributions as features for a regression model to determine whether or not that given commodity will go up or down. 

This GitHub repo has all headlines published from Reuters between 2007 and 2016 in a pickled file format. These are then unserialized and then converted from a file corpus to a line corpus (stored as single CSV file). These headlines are filtered for the occurence of the word 'oil'. 

The source of data is from this GitHub Repo : https://github.com/philipperemy/Reuters-full-data-set


### Getting Setup

If you haven't already please install mallet. There is also an alternative LDA model we will allow usage of (LDA Multicore). 

https://programminghistorian.org/en/lessons/topic-modeling-and-mallet#installing-mallet

### Usage

To begin with CTADaily please navigate to the IPython Notebook, which can be run cell by cell to walkthrough topic modelling and implementation/optimisation techniques. 

-  CTADaily.ipynb	(Notebook To Run)
-  CTADaily.py	(Contains Helper Methods)
-  DCOILWTICO.csv (Data Extract from FRED)
-  WTI-LDA-TOPIC.csv	(LDA Model Saved)
-  WTI-NEWS.csv (Reuters Headlines - Line Corpus)
