'''This program splits the original dataset to a separate datasets that only contains the 14 characters.'''
import pandas as pd
import csv

df = pd.read_csv('./Datasets/Simpson/simpsons_script_lines.csv', error_bad_lines=False, warn_bad_lines=False, low_memory=False)
df.dropna(inplace=True)

character = [2, 1, 8, 9, 15, 17, 3, 11, 25, 31, 139, 71, 165, 101] # Character to be kept

file = open('./14Character.csv', 'w')
writer = csv.writer(file)
# Write the first attribute row
writer.writerow(["id", "episode_id", "number", "raw_text", "timestamp_in_ms", "speaking_line", "character_id", "location_id", "raw_character_text", "raw_location_text", "spoken_words", "normalized_text", "word_count"])

for i in range(len(df)):
	cid = int(df.iloc[i].loc["character_id"])
	if cid in character:
		print(i)
		writer.writerow([
				df.iloc[i].loc["id"], 
				df.iloc[i].loc["episode_id"], 
				df.iloc[i].loc["number"], 
				df.iloc[i].loc["raw_text"], 
				df.iloc[i].loc["timestamp_in_ms"], 
				df.iloc[i].loc["speaking_line"], 
				df.iloc[i].loc["character_id"], 
				df.iloc[i].loc["location_id"], 
				df.iloc[i].loc["raw_character_text"], 
				df.iloc[i].loc["raw_location_text"], 
				df.iloc[i].loc["spoken_words"], 
				df.iloc[i].loc["normalized_text"], 
				df.iloc[i].loc["word_count"]])
	i+=1