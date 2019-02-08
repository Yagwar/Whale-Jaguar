import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

'''
Este script toma las frecuencias de las palabras y las suma a los autores para identificar el estilo de escritura.
Se eliminan las primeras 150 (por ser  en extremo frecuentes, se toman como stopwords)
Se utilizan las 500 palabras más frecuentes
'''

blog_data=pd.read_csv("blogtext.csv")

blog_data.drop_duplicates(subset="text",inplace=True)
blog_data.date = pd.to_datetime(blog_data.date,format="%d,%B,%Y", errors='coerce')
print("Blog Data Loaded")
print(blog_data.shape)

count_vect = CountVectorizer(max_features=150+500)
n_grams_counts = count_vect.fit_transform(blog_data["text"])#.toarray()

print("TF matrix done")
n_grams_counts.shape

txt_info= n_grams_counts.toarray().sum(axis=0)
indices = np.argsort(txt_info)[::-1]

freq_filt=n_grams_counts[:,indices[150:]]
print("Filtered TF matrix: ",freq_filt.shape)

df_out=pd.SparseDataFrame(freq_filt,
                   columns=np.array(count_vect.get_feature_names())[indices[150:]],
                   index=list(blog_data.index))
df_out.fillna(0,inplace=True)

# print("exporting TF matrix")
# try:
#     df_out.to_csv("df_freq_out.csv")
# except Exception as e:
#     print("type error: " + str(e))

    
print("Grouping TF data by author")
try:
    usr_txt_frq=df_out.groupby(blog_data["id"],sort=False).sum()
except Exception as e:
    print("type error: " + str(e))

print("Exporting data")
try:
    usr_txt_frq.to_csv("usr_txt_wrd_frqs.csv")
except Exception as e:
    print("type error: " + str(e))

    
print("Process Finished")