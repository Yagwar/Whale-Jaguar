import csv
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
df_out=df_out.to_dense()
df_out=df_out.apply(pd.to_numeric,downcast='unsigned')
print("TF DataFrame complete")

print("Exporting words")
with open('frq_outh_complete_colnames.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(list(df_out.columns))

# auth info
auths_ids=blog_data.id

print("Grouping TF data by author")
fr_auths=[list(df_out.loc[auths_ids==auth_id,].sum(axis=0)) for auth_id in list(set(auths_ids))]

# freqs_auths=pd.DataFrame(fr_auths,
#                          index=list(set(auths_ids)),
#                          columns=df_out.columns)

# freqs_auths=freqs_auths.apply(pd.to_numeric,downcast='unsigned')
print("Exporting data")
# try:
#     usr_txt_frq.to_csv("usr_txt_wrd_frqs.csv")
# except Exception as e:
#     print("type error: " + str(e))

with open('frq_outh_complete_1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(fr_auths)

print("Process Finished")
# exit()