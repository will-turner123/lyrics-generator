import requests
from rauth import OAuth1Service
import json
import pandas as pd
import numpy as np
import lyricsgenius as lg
import re
import os
from textgenrnn import textgenrnn
from keras.utils.np_utils import to_categorical
import yaml

# tensorflow==2.3.2


def get_artists():
    f = open('artists.txt')
    artists = []
    for artist in f.readlines():
        artist = artist.replace('\n', '')
        artists.append(artist)
    return artists



def get_lyrics(artists, max_song, genius):
    counter = 0
    rows = []
    lyrics_list = []
    for artist in artists:
        try:
            songs = (genius.search_artist(artist, max_songs=max_song, sort='popularity')).songs
            for song in songs:
                lyrics = song.lyrics
                # handle lyrics
                lyrics = lyrics.lower()
                #  embed stuff ends up in lyrics
                lyrics = lyrics.replace('embedshare urlcopyembedcopy', '')
                song_title = re.findall('"([^"]*)"', str(song))[0]
                rows.append([song_title, lyrics])
                lyrics_list.append(lyrics)
                # print(f'{song_title}: {lyrics}')
                counter += 1
        except Exception as e:
            print(f'Encountered exception on {artist} song #{counter}: {e}')
    df = pd.DataFrame(rows, columns=['song', 'lyrics'])
    return df, lyrics_list


def write_lyrics(lyrics, artists, i=1):
    with open(f'datasets/{i}.txt', 'w') as f:
        for item in lyrics:
            try:
                f.write(f'{item}\n')
            except UnicodeEncodeError:
                pass
    with open(f'mappings.json', 'r') as maps:
        maps = maps.read()
        try:
            maps = json.loads(maps)
        except:
            maps = {}
        try:
            maps[i] = {'artists': [artists], 'dataset': None, 'epochs': 0}
        except KeyError:
            maps = {i: {'artists': [artists], 'dataset': None, 'epochs': 0}}
        output_file = open('mappings.json', 'w')
        output_file.write(json.dumps(maps))


def log_dataset(i):
    config = read_config()
    with open(f'mappings.json', 'r') as maps:
        maps = maps.read()
        try:
            maps = json.loads(maps)
        except:
            maps = {}
        maps[str(i)]['dataset'] = i
        maps[str(i)]['epochs'] = config['epochs']
    output_file = open('mappings.json', 'w')
    output_file.write(json.dumps(maps))


def read_config():
    with open('config.yml', 'r') as f:
        conf = yaml.load(f.read())
        return conf

def read_lyrics(i=1):
    raw_text = open(f'datasets/{i}.txt', 'r').read()
    return raw_text

    
def write_output(output, id_counter=1):
    file_name = f'{id_counter}.txt'
    has_file = False
    file_counter = 0
    for file in os.listdir('outputs'):
        if file == file_name:
            has_file = True
    while has_file:
        file_name = f'{id_counter}.{file_counter}.txt'
        if file_name in os.listdir('outputs'):
            file_counter += 1
        else:
            has_file = False
    f = open(f'outputs/{file_name}', 'w')
    output = '\n'.join(output)
    f.write(output)


def do_model(id_counter=1, new_model=True):
    lyrics = read_lyrics(id_counter)
    config = read_config()
    if new_model:
        train_file = f'datasets/{id_counter}.txt'
        textgen = textgenrnn()
        textgen.train_from_file(train_file, num_epochs=config['epochs'])
        output = textgen.generate(int(config['song_length']), return_as_list=True)
        write_output(output, id_counter=id_counter)
        textgen.save(f'models/{id_counter}.hdf5')
        log_dataset(id_counter)
    else:
        train_file = f'models/{id_counter}.hdf5'
        textgen = textgenrnn(train_file)
        output = textgen.generate(int(config['song_length']), return_as_list=True)
        write_output(output, id_counter=id_counter)


def get_new_id():
    counter = 0
    for file in os.listdir('datasets'):
        counter += 1
    return counter + 1


def do_lyrics():
    config = read_config()
    genius = lg.Genius(config['access_token'], skip_non_songs=True, remove_section_headers=True)
    artists = get_artists()
    new_id = get_new_id()
    print(f'Going to read {config["scraped_songs_quota"]} songs from each artist...')
    df, lyrics = get_lyrics(artists, config['scraped_songs_quota'], genius) # list of artists, maximum songs to get, genius wrapper
    write_lyrics(lyrics, artists, new_id)
    return new_id



def option_one():
    new_id = do_lyrics()
    do_model(new_id, True)

def option_two():
    try:
        mappings = open('mappings.json', 'r').read()
        mappings = json.loads(mappings)
    except:
        print(f'Looks like you have no models! Try running the program again with option 1')
        return
    print(f'*' * 30)
    print(f'Your models: ')
    for key in mappings.keys():
        print(f' * {key}: Artists: {mappings[key]["artists"][0]}')
        print(f'       Epochs: {mappings[key]["epochs"]}')
    print('\n')
    model_select = input('Select which model you would like to generate lyrics with')
    if model_select in mappings:    
        do_model(id_counter=int(model_select), new_model=False)
    else:
        print(f'Unrecognized model. Run the program again!')


def main():
    print('\n' * 5)
    print(f'*' * 30)
    print("Lyrics Generator")
    print(F'*' * 30)
    print('\n' * 2)
    print(f'What would you like to do? Enter 1 or 2')
    print(f'1. Get lyrics from artists.txt, train new model on it and generate lyrics')
    print(f'2. Generate lyrics from an already trained model')
    user_input = input()
    if user_input != '1' and user_input != '2':
        print(f'Unrecognized input. Try again!')
        main()
    elif user_input == '1':
        option_one()
        return
    elif user_input == '2':
        option_two()
        return

main()
