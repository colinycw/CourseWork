'''This program splits 14Character.csv into training and testing data.'''
from sklearn.model_selection import train_test_split
import pandas as pd
import csv

df = pd.read_csv('./14Character.csv', error_bad_lines=False, warn_bad_lines=False, low_memory=False)
df.dropna(inplace=True)
df.drop(['episode_id', 'number', 'raw_text', 'timestamp_in_ms', 'location_id', 'raw_location_text', 'word_count'], axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(df[['id', 'speaking_line', 'raw_character_text', 'spoken_words' , 'normalized_text']], df['character_id'], test_size=0.3)

X_train.to_csv('./X_train.csv',index=True,header=True)
y_train.to_csv('./y_train.csv',index=True,header=True)

X_test.to_csv('./X_test.csv',index=True,header=True)
y_test.to_csv('./y_test.csv',index=True,header=True)