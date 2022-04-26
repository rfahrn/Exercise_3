#!/usr/bin/env python3
# *-* coding: UTF-8 *-*
# Authors: Jessica Roady & Rebecka Fahrni

from bertopic import BERTopic
import pandas as pd

# Change to nrows=5563 later
df = pd.read_json('https://files.ifi.uzh.ch/cl/siclemat/lehre/fs21/tm/data/all_de_topics.jsonl', lines=True, nrows=3000)
docs = df.iloc[:,0].tolist()

topic_model = BERTopic(verbose=True, language='German', nr_topics='auto')
topics, probs = topic_model.fit_transform(docs)

# TODO: Save the model here once you're satisfied with it.

print(topic_model.get_topic_info())

# Lots of stopwords. Let's try increasing the data from 1000 to 3000.
# If this doesn't help, let's try the filter_extremes() thing Simon mentioned, then CountVectorizer()
# okay maybe the id2word thing is specific to LDA, let's try nr_topics='auto'
