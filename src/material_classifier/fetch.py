from google_images_download import google_images_download  # importing the library
import os.path as p
import csv


def fetch_images(search, number, dir_name='google_downloads'):
    """
    Fetches number amount of images from Google Images starting from the image result 0.

    :param search: search query
    :param number: number of images. If >100, the chromedriver.exe has to be added to the PATH and Google Chrome must be installed.
    :param dir_name: name of the directory
    :return: image names
    """
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": search,
                 "limit": number,
                 "output_directory": p.split(dir_name)[0],
                 "image_directory": p.split(dir_name)[1],
                 "chromedriver": p.split(p.abspath(__file__))[0] + '/chromedriver.exe'}
    paths = response.download(arguments)
    return paths


def search_and_label(labels, number=5, dir_name='google_downloads', csv_file='./data.csv'):
    """
    Searches the different labels and downloads the same amount for every label.

    :param labels: labels
    :param number: numbers
    :param dir_name: directory name
    :param csv_file: path to the csv
    """
    k = 0
    for label in labels:
        paths = fetch_images(search=label, number=number, dir_name=dir_name)

        vals = list(paths[0].values())[0]
        for path in vals:
            row = [{'image': p.split(path)[1], 'labeled': k}]

            with open(csv_file, 'a', newline='') as outfile:
                fieldnames = ['image', 'labeled']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
                writer.writerows(row)

        k += 1

