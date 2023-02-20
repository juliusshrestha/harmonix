import pandas as pd
from yt_dlp import YoutubeDL

#Download the video 
def download_video(youtube_data):
   for index, files, urls in youtube_data.itertuples():
   
      ydl_opts = {
         'format': 'best',
         'outtmpl': 'D:\Projects\music\Music_Seg\harmonix\\Dataset\\Video\\'+files+'.mp4',
         'ignoreerrors': True,
         'geo_bypass': True,
         'geo_bypass_country': 'NP',
            }

      URLS = [urls]
      with YoutubeDL(ydl_opts) as ydl:
         ydl.download(URLS)

#Download the audio
def download_audio(youtube_data):
   for index, files, urls in youtube_data.itertuples():
   
      ydl_opts = {
         'format': 'bestaudio/best',
         'outtmpl': 'D:\Projects\music\Music_Seg\harmonix\\Dataset\\Audio\\'+files+'.mp3',
         'ignoreerrors': True,
         'geo_bypass': True,
         'geo_bypass_country': 'NP',
         'postprocessors': [{
            'key': 'FFmpegExtractAudio',
                  'preferredcodec': 'mp3',
            }]}
            

      URLS = [urls]
      with YoutubeDL(ydl_opts) as ydl:
         ydl.download(URLS)

if __name__ == "__main__":
    youtube_data = pd.read_csv("D:\Projects\music\Music_Seg\harmonix\youtube_urls.csv")
    download_video(youtube_data)
    download_audio(youtube_data)
    print("Downloaded!!")
    

