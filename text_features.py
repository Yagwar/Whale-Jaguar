import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import random
import time
import matplotlib.animation as animation
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
# from wordcloud import WordCloud
from sklearn.feature_extraction import text

from mpl_toolkits.mplot3d import Axes3D

blog_data=pd.read_csv("blogtext.csv")

blog_data.drop_duplicates(subset="text",inplace=True)
blog_data.date = pd.to_datetime(blog_data.date,format="%d,%B,%Y", errors='coerce')#pd.to_datetime(blog_data.date,errors="coerce",infer_datetime_format=True)
print("Blog Data Loaded")
print(blog_data.shape)
# blog_data.sample(20)

adit_stpwrds=["urllink","nbsp","ve","ll"]
stp_wrds = text.ENGLISH_STOP_WORDS.union(adit_stpwrds)

tfidf_transformer = TfidfVectorizer(stop_words=stp_wrds,max_features=500 )
n_grams_tfidf = tfidf_transformer.fit_transform(blog_data["text"])
print("TF-IDF matrix done")
n_grams_tfidf.shape

df_out=pd.DataFrame(blog_data["id"]).join(pd.SparseDataFrame(n_grams_tfidf,
                                         columns=tfidf_transformer.get_feature_names(),
                                         index=list(blog_data.index)))#.to_csv("tfidf_pd_texts.csv")

# print("exporting tf idf matrix")
# try:
#     df_out.to_csv("df_out.csv")
# except Exception as e:
#     print("type error: " + str(e))


print("Grouping tf-idf data by author")
try:
    usr_txt_ftr=df_out.groupby(['id'], sort=False).sum()
except Exception as e:
    print("type error: " + str(e))

print("Exporting data")
try:
    usr_txt_ftr.to_csv("usr_txt_ftr.csv")
except Exception as e:
    print("type error: " + str(e))
