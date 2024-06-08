"""
This produces file contains functions to produce random images of buildings
 from random cordinates in Zurich. 

It was used for experimentation and development but is not used in the final pipeline.
"""

import google_streetview.api
import tempfile
from PIL import Image
import random as rand

class randImgs:
    """Current generates a random image within a square grid of zurich"""
    def __init__(self):
        self.visited = set()

    def randCoordGen(self):
        """returns a random coordinate generator"""
        lowerR = 47.365759, 8.573775
        upperL = 47.386149, 8.491742

        #round to 6 digits after decimal and save so we don't visit same location twice
        while(True):
            lat = round(lowerR[0] + (rand.random() * (upperL[0]-lowerR[0])), 6)
            lon = round(upperL[1] + (rand.random() * (lowerR[1] - upperL[1])),6)
            if (lat,lon) not in self.visited: break
        
        self.visited.add((lat,lon))

        print("Generated: (%f, %f)"%(lat,lon))
        
        yield lat,lon

    def randCoord(self):
        """ returns a random coordinate"""
        return next(self.randCoordGen())

    def randImg(self):
        """Generates a random 600x600 img in the current directory title 'gcv_0.jpg'"""
        c = self.randCoord()
        params = [{
                'size': '600x600',  # max 640x640 pixels
                'location': str(c[0]) + ',' + str(c[1]),
                'heading': 0,
                'pitch': '25',
                # 'fov': '0',
                'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
            }]  
        results = google_streetview.api.results(params)
        results.download_links("./")

if  __name__ == "__main__":
    r = randImgs()
    r.randImg()
    # params = [{
    #             'size': '600x600',  # max 640x640 pixels
    #             'location': str(47.371656) + ',' + str(8.535608),
    #             'heading': 0,
    #             'pitch': '25',
    #             # 'fov': '0',
    #             'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
    #         }]  
    # results = google_streetview.api.results(params)
    # results.download_links("./")

    # params = [{
    #             'pano': 'ZLDhdGbaxaiEjhsOQZDBiQ',
    #             # 'size': '600x600',  # max 640x640 pixels
    #             # 'location': str(47.371590) + ',' + str(8.535710),
    #             # 'heading': 0,
    #             # 'pitch': '25',
    #             # # 'fov': '0',
    #             'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
    #         }]  
    # results = google_streetview.api.results(params)
    # results.download_links("./a")