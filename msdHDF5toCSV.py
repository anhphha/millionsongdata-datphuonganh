"""
Alexis Greenstreet (October 4, 2015) University of Wisconsin-Madison

This code is designed to convert the HDF5 files of the Million Song Dataset
to a CSV by extracting various song properties.

The script writes to a "SongCSV.csv" in the directory containing this script.

Please note that in the current form, this code only extracts the following
information from the HDF5 files:
AlbumID, AlbumName, ArtistID, ArtistLatitude, ArtistLocation,
ArtistLongitude, ArtistName, Danceability, Duration, KeySignature,
KeySignatureConfidence, SongID, Tempo, TimeSignature,
TimeSignatureConfidence, Title, and Year.

This file also requires the use of "hdf5_getters.py", written by
Thierry Bertin-Mahieux (2010) at Columbia University

Credit:
This HDF5 to CSV code makes use of the following example code provided
at the Million Song Dataset website
(Home>Tutorial/Iterate Over All Songs,
http://labrosa.ee.columbia.edu/millionsong/pages/iterate-over-all-songs),
Which gives users the following code to get all song titles:
"""
# import os
# import glob
# import hdf5_getters
# def get_all_titles(basedir,ext='.h5') :
#     titles = []
#     for root, dirs, files in os.walk(basedir):
#         files = glob.glob(os.path.join(root,'*'+ext))
#         for f in files:
#             h5 = hdf5_getters.open_h5_file_read(f)
#             titles.append( hdf5_getters.get_title(h5) )
#             h5.close()
#     return titles

import sys
import os
import glob
import hdf5_getters
import re

class Song:
    songCount = 0
    # songDictionary = {}

    def __init__(self, songID):
        self.id = songID
        Song.songCount += 1
        # Song.songDictionary[songID] = self

        self.albumName = None
        self.albumID = None
        self.artistID = None
        self.artistName = None
        self.duration = None
        self.genreList = []
        self.keySignature = None
        self.keySignatureConfidence = None
        self.lyrics = None
        self.popularity = None
        self.tempo = None
        self.timeSignature = None
        self.timeSignatureConfidence = None
        self.title = None
        self.year = None
        self.songHotness = None
        self.loudness = None
        self.audioMd5 = None
        # self.bars_start = []
        # self.beats_start = []
        # self.bars_confidence = []
        # self.beats_confidence = []
        # self.sections_start = None
        # self.sections_confidence = None
        # self.segments_start = None
        # self.segments_loudness_start = None
        # self.segments_loudness_max = None
        # self.segments_confidence = None
        # self.segments_loudness_max_time = None
        # self.segments_pitches = None
        # self.segments_timbre = None
        # self.tatums_confidence = None
        # self.tatums_start = None
        self.mode = None
        self.mode_confidence = None
        self.artist_hotttnesss = None
        # self.artist_terms_freq = []
        # self.artist_terms = None
        # self.artist_terms_weight = None

    # def displaySongCount(self):
    #     print "Total Song Count %i" % Song.songCount
    #
    # def displaySong(self):
    #     print "ID: %s" % self.id


def main():
    outputFile1 = open('SongCSV.csv', 'w')
    csvRowString = ""

    #################################################
    #if you want to prompt the user for the order of attributes in the csv,
    #leave the prompt boolean set to True
    #else, set 'prompt' to False and set the order of attributes in the 'else'
    #clause
    prompt = False
    #################################################
    if prompt == True:
        while prompt:

            prompt = False

            csvAttributeString = raw_input("\n\nIn what order would you like the colums of the CSV file?\n" +
                "Please delineate with commas. The options are: " +
                "ArtistID,"+
                " ArtistName, Duration, KeySignature, KeySignatureConfidence, Tempo," +
                " SongID, TimeSignature, TimeSignatureConfidence, Title, and Year.\n\n" +
                "For example, you may write \"Title, Tempo, Duration\"...\n\n" +
                "...or exit by typing 'exit'.\n\n")

            csvAttributeList = re.split('\W+', csvAttributeString)
            for i, v in enumerate(csvAttributeList):
                csvAttributeList[i] = csvAttributeList[i].lower()

            for attribute in csvAttributeList:
                # print "Here is the attribute: " + attribute + " \n"


                if attribute == 'ArtistID'.lower():
                    csvRowString += 'ArtistID'
                elif attribute == 'ArtistName'.lower():
                    csvRowString += 'ArtistName'
                elif attribute == 'Duration'.lower():
                    csvRowString += 'Duration'
                elif attribute == 'KeySignature'.lower():
                    csvRowString += 'KeySignature'
                elif attribute == 'KeySignatureConfidence'.lower():
                    csvRowString += 'KeySignatureConfidence'
                elif attribute == 'SongID'.lower():
                    csvRowString += "SongID"
                elif attribute == 'Tempo'.lower():
                    csvRowString += 'Tempo'
                elif attribute == 'TimeSignature'.lower():
                    csvRowString += 'TimeSignature'
                elif attribute == 'TimeSignatureConfidence'.lower():
                    csvRowString += 'TimeSignatureConfidence'
                elif attribute == 'Title'.lower():
                    csvRowString += 'Title'
                elif attribute == 'Year'.lower():
                    csvRowString += 'Year'
                elif attribute == 'SongHotness'.lower():
                    csvRowString += 'SongHotness'
                elif attribute == 'Loudness'.lower():
                    csvRowString += 'Loudness'
                elif attribute == 'AudioMd5'.lower():
                    csvRowString += 'AudioMd5'
                # elif attribute == 'bars_start'.lower():
                #     csvRowString += 'bars_start'
                # elif attribute == 'beats_start'.lower():
                #     csvRowString += 'beats_start'
                # elif attribute == 'bars_confidence'.lower():
                #     csvRowString += 'bars_confidence'
                # elif attribute == 'beats_confidence'.lower():
                #     csvRowString += 'beats_confidence'
                # elif attribute == 'sections_start'.lower():
                #     csvRowString += 'sections_start'
                # elif attribute == 'sections_confidence'.lower():
                #     csvRowString += 'sections_confidence'
                # elif attribute == 'segments_start'.lower():
                #     csvRowString += 'segments_start'
                # elif attribute == 'segments_loudness_start'.lower():
                #     csvRowString += 'segments_loudness_start'
                # elif attribute == 'segments_loudness_max'.lower():
                #     csvRowString += 'segments_loudness_max'
                # elif attribute == 'segments_confidence'.lower():
                #     csvRowString += 'segments_confidence'
                # elif attribute == 'segments_loudness_max_time'.lower():
                #     csvRowString += 'segments_loudness_max_time'
                # elif attribute == 'segments_pitches'.lower():
                #     csvRowString += 'segments_pitches'
                # elif attribute == 'segments_timbre'.lower():
                #     csvRowString += 'segments_timbre'
                # elif attribute == 'tatums_confidence'.lower():
                #     csvRowString += 'tatums_confidence'
                # elif attribute == 'tatums_start'.lower():
                #     csvRowString += 'tatums_start'
                elif attribute == 'mode'.lower():
                    csvRowString += 'mode'
                elif attribute == 'mode_confidence'.lower():
                    csvRowString += 'mode_confidence'
                elif attribute == 'artist_hotttnesss'.lower():
                    csvRowString += 'artist_hotttnesss'
                # elif attribute == 'artist_terms_freq'.lower():
                #     csvRowString += 'artist_terms_freq'
                # elif attribute == 'artist_terms'.lower():
                #     csvRowString += 'artist_terms'
                # elif attribute == 'artist_terms_weight'.lower():
                #     csvRowString += 'artist_terms_weight'
                elif attribute == 'Exit'.lower():
                    sys.exit()
                else:
                    prompt = True
                    print ("==============")
                    print ("I believe there has been an error with the input.")
                    print ("==============")
                    break

                csvRowString += ","

            lastIndex = len(csvRowString)
            csvRowString = csvRowString[0:lastIndex-1]
            csvRowString += "\n"
            outputFile1.write(csvRowString);
            csvRowString = ""
    #else, if you want to hard code the order of the csv file and not prompt
    #the user,
    else:
        #################################################
        #change the order of the csv file here
        #Default is to list all available attributes (in alphabetical order)
        csvRowString = ("SongID,ArtistID,"+
            "ArtistName,Duration,KeySignature,"+
            "KeySignatureConfidence,Tempo,TimeSignature,TimeSignatureConfidence,"+
            "Title,Year,SongHotness,Loudness,AudioMd5,"+
            # "bars_start,beats_start,bars_confidence,beats_confidence,sections_start,sections_confidence,"+
            # "segments_start,segments_loudness_start,segments_loudness_max,segments_confidence,segments_loudness_max_time,segments_pitches,"+
            # "segments_timbre,tatums_confidence,tatums_start,"+
            "mode,mode_confidence,artist_hotttnesss")

        #################################################

        csvAttributeList = re.split('\W+', csvRowString)
        for i, v in enumerate(csvAttributeList):
            csvAttributeList[i] = csvAttributeList[i].lower()
        outputFile1.write("SongNumber,");
        outputFile1.write(csvRowString + "\n");
        csvRowString = ""

    #################################################


    #Set the basedir here, the root directory from which the search
    #for files stored in a (hierarchical data structure) will originate
    basedir = "/scratch/project_2000859/datasets/One_Million_Songs/" # "." As the default means the current directory
    ext = ".h5" #Set the extension here. H5 is the extension for HDF5 files.
    #################################################

    #FOR LOOP
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            print(f)

            songH5File = hdf5_getters.open_h5_file_read(f)
            song = Song(str(hdf5_getters.get_song_id(songH5File)))

            testDanceability = hdf5_getters.get_danceability(songH5File)
            # print type(testDanceability)
            # print ("Here is the danceability: ") + str(testDanceability)

            song.artistID = str(hdf5_getters.get_artist_id(songH5File))
            song.artistName = str(hdf5_getters.get_artist_name(songH5File))
            song.duration = str(hdf5_getters.get_duration(songH5File))
            # song.setGenreList()
            song.keySignature = str(hdf5_getters.get_key(songH5File))
            song.keySignatureConfidence = str(hdf5_getters.get_key_confidence(songH5File))
            # song.lyrics = None
            # song.popularity = None
            song.tempo = str(hdf5_getters.get_tempo(songH5File))
            song.timeSignature = str(hdf5_getters.get_time_signature(songH5File))
            song.timeSignatureConfidence = str(hdf5_getters.get_time_signature_confidence(songH5File))
            song.title = str(hdf5_getters.get_title(songH5File))
            song.year = str(hdf5_getters.get_year(songH5File))
            song.songHotness = str(hdf5_getters.get_song_hotttnesss(songH5File))
            song.loudness = str(hdf5_getters.get_loudness(songH5File))
            song.audioMd5 = str(hdf5_getters.get_audio_md5(songH5File))
            # song.bars_start = str(hdf5_getters.get_bars_start(songH5File))
            # song.beats_start = str(hdf5_getters.get_beats_start(songH5File))
            # song.bars_confidence = str(hdf5_getters.get_bars_confidence(songH5File))
            # song.beats_confidence = str(hdf5_getters.get_beats_confidence(songH5File))
            # song.sections_start = str(hdf5_getters.get_sections_start(songH5File))
            # song.sections_confidence = str(hdf5_getters.get_sections_confidence(songH5File))
            # song.segments_start = str(hdf5_getters.get_segments_start(songH5File))
            # song.segments_loudness_start = str(hdf5_getters.get_segments_loudness_start(songH5File))
            # song.segments_loudness_max = str(hdf5_getters.get_segments_loudness_max(songH5File))
            # song.segments_confidence = str(hdf5_getters.get_segments_confidence(songH5File))
            # song.segments_loudness_max_time = str(hdf5_getters.get_segments_loudness_max_time(songH5File))
            # song.segments_pitches = str(hdf5_getters.get_segments_pitches(songH5File))
            # song.segments_timbre = str(hdf5_getters.get_segments_timbre(songH5File))
            # song.tatums_confidence = str(hdf5_getters.get_tatums_confidence(songH5File))
            # song.tatums_start = str(hdf5_getters.get_tatums_start(songH5File))
            song.mode = str(hdf5_getters.get_mode(songH5File))
            song.mode_confidence = str(hdf5_getters.get_mode_confidence(songH5File))
            song.artist_hotttnesss = str(hdf5_getters.get_artist_hotttnesss(songH5File))
            # song.artist_terms_freq = str(hdf5_getters.get_artist_terms_freq(songH5File))
            # song.artist_terms = str(hdf5_getters.get_artist_terms(songH5File))
            # song.artist_terms_weight = str(hdf5_getters.get_artist_terms_weight(songH5File))

            #print song count
            csvRowString += str(song.songCount) + ","

            for attribute in csvAttributeList:
                # print "Here is the attribute: " + attribute + " \n"


                if attribute == 'ArtistID'.lower():
                    csvRowString += "\"" + song.artistID + "\""
                elif attribute == 'ArtistName'.lower():
                    csvRowString += "\"" + song.artistName + "\""
                elif attribute == 'Duration'.lower():
                    csvRowString += song.duration
                elif attribute == 'KeySignature'.lower():
                    csvRowString += song.keySignature
                elif attribute == 'KeySignatureConfidence'.lower():
                    # print "key sig conf: " + song.timeSignatureConfidence
                    csvRowString += song.keySignatureConfidence
                elif attribute == 'SongID'.lower():
                    csvRowString += "\"" + song.id + "\""
                elif attribute == 'Tempo'.lower():
                    # print "Tempo: " + song.tempo
                    csvRowString += song.tempo
                elif attribute == 'TimeSignature'.lower():
                    csvRowString += song.timeSignature
                elif attribute == 'TimeSignatureConfidence'.lower():
                    # print "time sig conf: " + song.timeSignatureConfidence
                    csvRowString += song.timeSignatureConfidence
                elif attribute == 'Title'.lower():
                    csvRowString += "\"" + song.title + "\""
                elif attribute == 'Year'.lower():
                    csvRowString += song.year
                elif attribute == 'SongHotness'.lower():
                    csvRowString += song.songHotness
                elif attribute == 'Loudness'.lower():
                    csvRowString += song.loudness
                elif attribute == 'AudioMd5'.lower():
                    csvRowString += song.audioMd5
                # elif attribute == 'bars_start'.lower():
                #     csvRowString += song.bars_start
                # elif attribute == 'beats_start'.lower():
                #     csvRowString += song.beats_start
                # elif attribute == 'bars_confidence'.lower():
                #     csvRowString += song.bars_confidence
                # elif attribute == 'beats_confidence'.lower():
                #     csvRowString += song.beats_confidence
                # elif attribute == 'sections_start'.lower():
                #     csvRowString += song.sections_start
                # elif attribute == 'sections_confidence'.lower():
                #     csvRowString += song.sections_confidence
                # elif attribute == 'segments_start'.lower():
                #     csvRowString += song.segments_start
                # elif attribute == 'segments_loudness_start'.lower():
                #     csvRowString += song.segments_loudness_start
                # elif attribute == 'segments_loudness_max'.lower():
                #     csvRowString += song.segments_loudness_max
                # elif attribute == 'segments_confidence'.lower():
                #     csvRowString += song.segments_confidence
                # elif attribute == 'segments_loudness_max_time'.lower():
                #     csvRowString += song.segments_loudness_max_time
                # elif attribute == 'segments_pitches'.lower():
                #     csvRowString += song.segments_pitches
                # elif attribute == 'segments_timbre'.lower():
                #     csvRowString += song.segments_timbre
                # elif attribute == 'tatums_confidence'.lower():
                #     csvRowString += song.tatums_confidence
                # elif attribute == 'tatums_start'.lower():
                #     csvRowString += song.tatums_start
                elif attribute == 'mode'.lower():
                    csvRowString += song.mode
                elif attribute == 'mode_confidence'.lower():
                    csvRowString += song.mode_confidence
                elif attribute == 'artist_hotttnesss'.lower():
                    csvRowString += song.artist_hotttnesss
                # elif attribute == 'artist_terms_freq'.lower():
                #     csvRowString += song.artist_terms_freq
                # elif attribute == 'artist_terms'.lower():
                #     csvRowString += song.artist_terms
                # elif attribute == 'artist_terms_weight'.lower():
                #     csvRowString += song.artist_terms_weight
                else:
                    csvRowString += "Erm. This didn't work. Error. :( :(\n"

                csvRowString += ","

            #Remove the final comma from each row in the csv
            lastIndex = len(csvRowString)
            csvRowString = csvRowString[0:lastIndex-1]
            csvRowString += "\n"
            outputFile1.write(csvRowString)
            csvRowString = ""

            songH5File.close()

    outputFile1.close()

main()
