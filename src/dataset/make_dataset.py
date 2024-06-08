"""
This includes the main function img_from_db which is used by the pipeline to create images.

It also includes helper functions for images stitching which can be used by adding 
sample = True to the list of args in run.py for the "args.sample" command, or by changing the default
to sample = True in the img_from_db function

imgs_copied.txt: used to store the amoung of images already pulled from the API
imgs_copied_stitched.txt: same function, but if you are pulling stitched images
"""
import shapefile
import google_streetview.api
import googlemaps
#from gmaps import Roads
import math
import os
import shutil

# from sample.py
import tempfile
from PIL import Image
import random as rand

# for image_stitching
from imutils import paths
import numpy as np
import argparse
import imutils
import os.path
from cv2 import cv2

this_path = ""

def get_sides(lat, long, init_heading):
    # Used for image stitching: Define parameters for street view api
    images = []
    steps = 3  # Correlates to the good area ratio
    imgCount = len([name for name in os.listdir(this_path + 'stitchUnlabel') if os.path.isfile(os.path.join(this_path + 'stitchUnlabel', name))]) - 1

    with tempfile.TemporaryDirectory() as tmpdirname:

        for i in range(steps):
            params = [{
                'size': '640x640',  # max 640x640 pixels
                'location': str(lat) + ',' + str(long),
                'heading': str(init_heading - 30 + i*90/steps),
                'pitch': '25',
                # 'fov': '0',
                'source': 'outdoor',
                'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
            }]
            print(params)
            results = google_streetview.api.results(params)
            #print(results.metadata)
            results.download_links(tmpdirname)

            if not os.path.exists(tmpdirname + '/gsv_0.jpg'):
                continue
            image = Image.open(tmpdirname + '/gsv_0.jpg')
            image.save(this_path + 'imgToStitch' + '/gsv' + str(i) + '.jpg')
            image.save(this_path + 'rawImages' + '/gsv' + str(imgCount*3 + i) + '.jpg')
            image.close()
            image = Image.open(this_path + 'imgToStitch/gsv' + str(i) + '.jpg')
            images.append(image)

def img_stitch():
    # Used for image stitching: grab the paths to the input images and initialize our images list
    imgPaths = sorted(list(paths.list_images(this_path + "imgToStitch")))
    imgs = []

    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    for imgPath in imgPaths:
        img = cv2.imread(imgPath)
        imgs.append(img)

    # initialize OpenCV's image stitcher object and then perform the image
    # stitching
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(imgs)

    # if the status is '0', then OpenCV successfully performed image
    # stitching
    if status == 0:
        # create a 10 pixel border surrounding the stitched image
        stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
            cv2.BORDER_CONSTANT, (0, 0, 0))

        # get number of images already in the ./stitchUnlabel directory
        imgCount = len([name for name in os.listdir(this_path + 'stitchUnlabel') if os.path.isfile(os.path.join(this_path + 'stitchUnlabel', name))]) - 1

        # write the output stitched image to disk (adding 1 to imgCount bc first image is gsv1 not gsv0)
        cv2.imwrite(this_path + 'stitchUnlabel' + '/gsv'+ str(imgCount+1) + '.jpg', stitched)

    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))

def img_from_db(numImgs,final_path="", path="./", stitch = False):
    """
    This is the only function being used by the pipeline: 
    generates an image for the ith entry in the zurich building database by
    snapping it's point to the nearest roadview and collecting the image from 
    google streetview
    """
    #print("generating",numImgs, "images in", final_path)
    global this_path
    this_path = path + "\\src\\dataset\\" if final_path!="" else path
    #print("this_path",this_path)
    
    imgs_seen = set()  #this only holds images seen in this batch of generation... not all batches

    dest_folder = 'stitchUnlabel' if stitch else 'Unlabel'

    if stitch:  #just locations.csv
        imgCount = len([name for name in os.listdir(this_path + dest_folder) if os.path.isfile(os.path.join(this_path + dest_folder, name))]) -1
    else: 
        if not os.path.exists(this_path + 'imgs_copied.txt'):
            raise Exception("Need " + this_path + "imgs_copied.txt or imgs_copied_stitched.txt file to keep track of how many are copied. \
                Initialize file with integer 0 if starting from beginning of database")
        file = open(this_path + 'imgs_copied.txt',"r")
        imgCount = int(file.readline())
        file.close()
        #locations.csv and metadata.json
        print("ImgCount is ",imgCount)

    shape = shapefile.Reader(this_path + "Zurich_location\data_w_Zurich_location.shp")
    params = [] # used if stich=false
    if os.path.exists(this_path + dest_folder+ "/locations.csv"):
        locFile = open(this_path + dest_folder + "/locations.csv","a")
    else:
        locFile = open(this_path + dest_folder+ "/locations.csv","w+")
        locFile.write("Building, Road, Heading, Errors\n")

    for i in range(imgCount,imgCount+numImgs):
        #i=-i #uncomment this to generate images starting from the end of the database
        print(i)
        #first feature of the shapefile
        feature = shape.shapeRecords()[i]
        first = feature.shape.__geo_interface__  

        #c is center of building
        c = first['coordinates']
        c = round(c[1],5),round(c[0],5) #it looks like google maps goes up to 5 decimal point precision 
        print ("from database:",c) # (GeoJSON format)

        param_road = [{"lat" : c[0], "lng" : c[1]}] # we store it as (lat,long)
        gmaps = googlemaps.Client('AIzaSyDXspVdNLu7T_U3-RxqxRryffP-_kc3b3k')
        # r is cordinate snapped to road
        a = gmaps.snap_to_roads(param_road) #a is temp variable
        #print("Snapped to Road:",a)
        if len(a)==0:
            print("Roads API returned empty result...")
            locFile.write(f"{c[0]} {c[1]}, 0 0, 0, 3\n")
            continue
        r = [ round(a[0]['location']['latitude'],5), round(a[0]['location']['longitude'],5)]
        print("from snapped to road:",r)
        try:
            heading = math.degrees(math.atan((c[0]-r[0])/(c[1]-r[1]))) #lat. is like y axis (horiz lines on vertical axis), long. is like x 
        except ZeroDivisionError:
            heading = 90 if c[0]-r[0] > 0 else 270

        if c[1]-r[1] < 0: # if x is negative then it's supposed to be in quadrants 2 or 3
            heading+=180
        heading = 360-heading + 90 #zero is north not east like unit circle
        print('heading:', heading)

        if stitch:
            get_sides(r[0], r[1], heading)
            img_stitch()
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                params = [{
                    'size': '640x640',  # max 640x640 pixels
                    'location': str(r[0]) + ',' + str(r[1]), #note json format returns long,lat backwards
                    'heading': heading,
                    'pitch': '25',
                    'source': 'outdoor',
                    'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
                }]
                print(params)
                results = google_streetview.api.results(params)
                print(results.metadata)
                
                results.download_links(tmpdirname)
                
                if not os.path.exists(tmpdirname + '/gsv_0.jpg'):
                    print("indoors... going to next")
                    locFile.write(f"{c[0]} {c[1]}, {r[0]} {r[1]}, {heading}, 1\n")
                    continue

                id = results.metadata[0]['pano_id']
                if id in imgs_seen:
                    print("deja le voir... going to next")
                    locFile.write(f"{c[0]} {c[1]}, {r[0]} {r[1]}, {heading}, 2\n")
                    continue
                
                imgs_seen.add(id)

                image = Image.open(tmpdirname + '/gsv_0.jpg')
                image.save(this_path + 'Unlabel/gsv_' + str(i) + '.jpg')
                image.close()


        #locFile.write('%d %d, %d %d, %d'.format(c[0],c[1],r[0],r[1],heading, 0 (indicating outdoors)))
        locFile.write(f"{c[0]} {c[1]}, {r[0]} {r[1]}, {heading}, 0\n")
    

    locFile.close()
    #i think 0 degrees is north. and it increases clockwise

    #this copies the files over to the new directory 
    if final_path!="":
        txtfile_dir = "imgs_copied_stitched.txt" if stitch else "imgs_copied.txt"
        if stitch:
            #if images are stiched we didn't open this file to read how many images are already copied
            if not os.path.exists(this_path + txtfile_dir):
                raise Exception("Need " + this_path + "imgs_copied.txt or imgs_copied_stitched.txt file to keep track of how many are copied. \
                    Initialize file with integer 0 if starting from beginning of database")
            copied_tracker= open(this_path + txtfile_dir,"r")
            num_copied = int(copied_tracker.readline())
            copied_tracker.close()
        else: #if images aren't stitched we already read this file to get imgCount
            num_copied = imgCount
        #print("COPIED SO FAR %d"%num_copied)

        src_files = os.listdir(this_path + dest_folder)
        #print(src_files)
        for file_name in src_files:
            full_file_name = os.path.join(this_path + dest_folder, file_name)
            if stitch: 
                img_num = int(file_name[3:-4]) if full_file_name[-4:]=='.jpg' else -1
            else:
                img_num = int(file_name[4:-4]) if full_file_name[-4:]=='.jpg' else -1
            #print("EXTENSION: ", full_file_name, " ", full_file_name[-4:], img_num)

            if os.path.isfile(full_file_name) and full_file_name[-4:]=='.jpg' and img_num >= num_copied: #make sure we only copy imgs
                shutil.copy(full_file_name, final_path)

        copied_tracker= open(this_path + txtfile_dir,"w")
        copied_tracker.write(str(imgCount+numImgs))
        copied_tracker.close()

if __name__ == "__main__":
    """ This main program was just used for testing the make_dataset functions
    """
    # img_from_db(200)
    # img_stitch()
    img_from_db(5) 

    #uncomment the code below if you want to look up just one image from GoogleAPI
    """
    c=(47.37593, 8.54457)
    params = [{
                'size': '600x600',  # max 640x640 pixels
                'location': str(c[0]) + ',' + str(c[1]),
                'heading': 270,
                'pitch': '25',
                # 'fov': '0',
                'key': 'AIzaSyAwTObZ9-lfhdw9blad_ce4SUX5PWsufTI'
            }]  
    results = google_streetview.api.results(params)
    results.download_links("./")
    """
        

"""
csv key:
Building, Road, Heading, Errors
Buidling: lat_building long_building
Road: lat_snappedtoRoad long_snappedtoRoad
Heading: degrees pointing from road to building with 0 facing north and going clockwise
Errors:
0 = no errors
1 = indoor pic returned from google api
2 = repeated pic in same batch (meaning if you call in --sample 200, in only checks for repeats within those 200)
3 = google roads api returned empty list? 
"""