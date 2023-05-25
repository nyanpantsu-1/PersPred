import numpy as np
import pandas as pd
import re
import googleapiclient.discovery

data=pd.read_csv("./Archive/mbti_1.csv")

nrow,ncol=data.shape

types = np.unique(np.array(data['type']))

total=data.groupby(['type'])

regex = r'(https?://)?(www\.)?(youtube\.com|youtu\.be)/(watch\?v=)?([A-Za-z0-9_-]{11})'
data['links'] = data['posts'].apply(lambda x: re.findall(regex, x))

def get_video_title(video_url):
    video_id = video_url[4]
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey='API ACCESS KEY PLACEHODER')
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    return response['items'][0]['snippet']['title']

title_data=pd.Series(""*nrow)
#print(data.links[2])



    






