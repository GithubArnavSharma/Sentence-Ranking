import numpy as np
import tensorflow_text as text
from tensorflow_hub import KerasLayer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8' 
handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

bert_model = KerasLayer(handle_encoder)
bert_preprocess = KerasLayer(handle_preprocess)

def vectorize_text(text):
  text = [t.lower() for t in text]
  processed_text = bert_preprocess(text)
  bert_output = bert_model(processed_text)['pooled_output']
  return bert_output

def text_similarity(a, b):
  texts = [a, b]
  vectors = vectorize_text(texts)
  text1_vector = [np.array(vectors[0]).reshape(512,)]
  text2_vector = [np.array(vectors[1]).reshape(512,)]
  sim = cosine_similarity(text1_vector, text2_vector)
  return sim

def ranked_text(other_texts, text):
  number_text = len(other_texts)
  similarities = [text_similarity(other_texts[i], text) for i in range(number_text)]

  text2sim = {other_texts[i]: similarities[i] for i in range(number_text)}
  text2sim = {text: sim for text, sim in sorted(text2sim.items(), key=lambda item: item[1])}
  return list(reversed(list(text2sim.keys())))
