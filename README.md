# SI507 Final Project: Spotify Playlist Analyzer

# About this Project

This is an interactive program using the Spotify, Wikipedia, and Genius Lyrics API that retrieves information about the Top 100 Songs of 2022 Spotify playlist. Instructions are provided below to successfully run the program.

# Data Sources

## I. Spotify API 
I used Spotipy, which is a Python library for the Spotify Web API, to retrieve information about songs in the playlist and its features. Here is the URL to the Spotipy documentation: https://spotipy.readthedocs.io/en/2.19.0/. 
To access the data, log into your Spotify account. Then, create an app. Make sure to take note of your client ID, access token, and client secret. 

## II. Wikipedia API
I also used the Wikipedia API for the purposes of retrieving information about a song. Here is the URL to the Wikipedia documentation: https://wikipedia.readthedocs.io/en/latest/code.html#api. 
API key is not required to retrieve data, so no additional steps are required.

## III. Genius Lyrics API
Another data source that was used was Genius Lyrics. This was necessary for retrieving lyrics for the top 10 songs in the playlist and performing sentiment analysis. Here is the URL to the Genius Lyrics documentation: https://docs.genius.com/#/search-h2. 
To access the data, log into your Genius account. Access the following link: https://genius.com/api-clients, and create a new API client. Fill out a short form to create an App. After creating an App, you should be given a client ID and client secret. Make sure to get a client access token, click on "Generate Access Token". For this program, only the access token is needed.

# List of Required Python Packages
<li>pandas</li>
<li>spotipy</li>
<li>seaborn</li>
<li>plotly</li>
<li>gensim</li>
<li>wordcloud</li>
<li>vaderSentiment</li>
<li>nltk</li>
<li>numpy</li>
<li>igraph</li>
<li>beautifulsoup4</li>
<li>tabulate</li>
<li>wikipedia</li>
<li>matplotlib</li>
<li>Counter</li>
<li>regex</li>
<li>urllib3</li>
<li>html5lib</li>
<li>tsne</li>
<li>scikit-learn</li>

# Data Structure

## Spotify Data

I created a SpotifyAPI class and used the spotipy library to retrieve information about the songs in the playlist. This class contained functions that were essential for getting the album, artist, release date, song features (which included length, popularity, acousticness, danceability, energy, key, mode, valence, instrumentalness, liveness, loudness, speechiness, tempo, time signature) and genre information about each song. 

The resulting data was cached as a json file and turned into a dataframe. The json file of this data is under 'spotifycache.json'. Below is a snippet of how the json file looks:

<img width="500" alt="Graph.png" src="https://github.com/eychoi218/SI507/blob/main/images/Spotify.png">

## Wikipedia Data

I created a function called wikisummary to retrieve information from Wikipedia for each song in the playlist. This function captured the following information: summary, title, URL, content, links, images on the Wikipedia page when querying by song name. 

The resulting data was cached as a json file and turned into a dataframe. The json file of this data is under 'wikicache.json'. Below is a snippet of how the json file looks:

<img width="500" alt="Graph.png" src="https://github.com/eychoi218/SI507/blob/main/images/Wiki.png">

## Genius Lyrics Data

I created a function called GetLyrics to retrieve lyrics from Genius for each song in the playlist.

The resulting data was cached as a json file and turned into a dataframe. The json file of this data is under 'geniuscache.json'. Below is a snippet of how the json file looks:

<img width="500" alt="Graph.png" src="https://github.com/eychoi218/SI507/blob/main/images/Genius.png">

## Graph Data Structure
Each artist could have more than one genre listed according to Spotify. Therefore, I have created a graph/network which provides information on genres and their degrees. For instance, whenever 2 genres have the same artist, an edge will be added to the graph. Furthermore, the graph will be weighted by the number of times genres are “tagged” together. 

Here is the information I provide using the graph:
<li>The number of unique genres in the playlist</li>
<li>Average degrees in the playlist</li>
<li>List of top 5 genres with the highest degree</li>
<li>List of top 5 genres with the highest weighted degree</li>

<br>The code for the graph exists within the spotify.py and spotify_eychoi.ipynb file. The json file of the graph is under 'graph.json'. A standalone Python file also exists under 'graph.py', which reads the json of the graph data structure and displays the graph figure. Below are images of how the json file for the graph data structure and graph figure look:</br>

<img width="500" alt="Graph.png" src="https://github.com/eychoi218/SI507/blob/main/images/Graph.png">

<img width="500" alt="Graph.png" src="https://github.com/eychoi218/SI507/blob/main/images/GraphFigure.png">

# Program Interaction
In general, the user can choose between 'yes' or 'no'. To select individual songs to analyze, the user will have the option to enter a number ranging from 1-10 or can choose to exit the program by typing 'exit'.

<ol type="1">
<li>When the user runs the program (either from the 'spotify.py' or 'spotify_eychoi.ipynb' file), a welcome message will display and ask if the user would like to see a summary of the TOP 100 Songs of 2022 (Best Hit Music Playlist) Spotify playlist.</li>
<li>The user will have the option to type ‘Yes’ or ‘No’. If the user types ‘No’, the message ‘Goodbye!’ will print. On the other hand, if the user types ‘Yes’, a summary of the playlist will be provided (list of popular songs; seaborn and plotly visualizations such as the radar chart of audio features, heatmap of audio features, most popular songs and genres, histograms of audio features, information about weighted degrees of top genres).</li>
<li>The user will then be prompted to answer ‘Yes’ or ‘No’ when asked if they would like to see the lyrics to the most popular songs. A ‘No’ answer will trigger the user to answer if they would like to see the summary playlist. A ‘Yes’ answer will re-provide the list of the most popular songs in the playlist and ask the user to select a number (which represents the songs in the list).
<li>Here, the user can either enter ‘exit’ to exit the program, which will trigger a ‘Goodbye!’ message or the user can enter a number from 1-10. Entering a number from 1-10, the user will be provided the lyrics for the chosen song, along with the sentiment analysis and a wordcloud showing the most important words from the lyrics.</li>
<li>Then, the user will be asked if they would like to see more information about the song in Wikipedia. A ‘Yes’ answer will provide the user with a summary from the Wikipedia page. A ‘No’ answer will redirect the user to choose another song from the list of popular songs to analyze.</li>
<li>After the user has chosen ‘Yes’, the user will also be asked if they would like the program to open the actual Wikipedia page. If the user says ‘Yes’, the webpage of the song will open in the browser. If the user says ‘No’, the user will be redirected to choose another song from the list of popular songs to analyze.</li>
<li>After, the user will automatically be asked to choose a song from the list of top songs. The user can either answer with a number from 1-10 or 'exit'.
</ol>
