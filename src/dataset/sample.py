"""
This file includes:
- basic image stitching functions: stitch, get_stitched_360, get_sides
- random cordinate generator within Zurich: random_coordinate()
none of these functions are used in the final pipeline.
"""

import google_streetview.api
import tempfile
from PIL import Image
import random as rand

def simple_un_fov(image):
    return image


def stitch(images, extra_lap=0):

    good_area = 1/20

    width, height = images[0].size

    good_width = int(width*good_area)
    total_width = int((len(images) + extra_lap)*good_width)
    left = int((width - good_width)/2)
    right = int((width + good_width)/2)

    for i in range(len(images)):
        #   (left, upper, right, lower)
        images[i] = simple_un_fov(images[i]).crop((left, 0, right, height))

    pos = 0
    dst = Image.new('RGB', (total_width, height))

    for img in images:
        dst.paste(img, (int(pos - 1), 0))
        pos += good_width

    for i in range(extra_lap):
        dst.paste(images[i], (int(pos - 1), 0))
        pos += good_width

    for image in images:
        image.close()

    return dst


def get_stitched_360(lat, long):
    # Define parameters for street view api
    images = []
    steps = 50  # Correlates to the good area ratio

    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(steps):
            params = [{
                'size': '600x600',  # max 640x640 pixels
                'location': str(lat) + ',' + str(long),
                'heading': str(i*360/steps),
                'pitch': '25',
                # 'fov': '0',
                'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
            }]
            print(params)
            results = google_streetview.api.results(params)

            results.download_links(tmpdirname)

            image = Image.open(tmpdirname + '/gsv_0.jpg')
            image.save('./results' + '/gsv' + str(i) + '.jpg')
            image.close()
            image = Image.open('./results/gsv' + str(i) + '.jpg')
            images.append(image)

        img = stitch(images, extra_lap=int(len(images)/4))
        img.show()
        img.save('./results/result.jpeg')


def get_sides(lat, long):
    # Define parameters for street view api
    images = []
    steps = 50  # Correlates to the good area ratio

    with tempfile.TemporaryDirectory() as tmpdirname:
        offset = 55
        side = 170
        road = 360-2*side

        first = side - offset
        second = first + road
        third = second + side
        fourth = third + road

        first  = int(steps / 360 * first) + 1
        second = int(steps / 360 * second) + 1
        third  = int(steps / 360 * third)
        fourth = int(steps / 360 * fourth)

        for i in range(fourth, steps, 1):
            params = [{
                'size': '600x600',  # max 640x640 pixels
                'location': str(lat) + ',' + str(long),
                'heading': str(i*360/steps),
                'pitch': '25',
                # 'fov': '0',
                'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
            }]
            print(params)
            results = google_streetview.api.results(params)

            results.download_links(tmpdirname)

            image = Image.open(tmpdirname + '/gsv_0.jpg')
            image.save('./results' + '/gsv' + str(i) + '.jpg')
            image.close()
            image = Image.open('./results/gsv' + str(i) + '.jpg')
            images.append(image)

        for i in range(first):
            params = [{
                'size': '600x600',  # max 640x640 pixels
                'location': str(lat) + ',' + str(long),
                'heading': str(i*360/steps),
                'pitch': '25',
                # 'fov': '0',
                'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
            }]
            print(params)
            results = google_streetview.api.results(params)

            results.download_links(tmpdirname)

            image = Image.open(tmpdirname + '/gsv_0.jpg')
            image.save('./results' + '/gsv' + str(i) + '.jpg')
            image.close()
            image = Image.open('./results/gsv' + str(i) + '.jpg')
            images.append(image)

        img = stitch(images)
        img.show()
        img.save('./results/side1.jpeg')

        images = []
        for i in range(second, third, 1):
            params = [{
                'size': '600x600',  # max 640x640 pixels
                'location': str(lat) + ',' + str(long),
                'heading': str(i*360/steps),
                'pitch': '25',
                # 'fov': '0',
                'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
            }]
            print(params)
            results = google_streetview.api.results(params)

            results.download_links(tmpdirname)

            image = Image.open(tmpdirname + '/gsv_0.jpg')
            image.save('./results' + '/gsv' + str(i) + '.jpg')
            image.close()
            image = Image.open('./results/gsv' + str(i) + '.jpg')
            images.append(image)

        img = stitch(images)
        img.show()
        img.save('./results/side2.jpeg')

def random_cordinate():
    #upperR = 47.391885, 8.558946
    lowerR = 47.365759, 8.573775
   # lowerL = 47.364528, 8.490702
    upperL = 47.386149, 8.491742

    randomlat = lowerR[0] + (rand.random() * (upperL[0]-lowerR[0]))
    randomlon = upperL[1] + (rand.random() * (lowerR[1] - upperL[1]))

    print(randomlat,randomlon)
    return randomlat,randomlon

    

if __name__ == "__main__":
    #random_cordinate()
    #get_sides(46.525762, 6.623534)
    c = random_cordinate()
    #get_sides(c[0],c[1])


            