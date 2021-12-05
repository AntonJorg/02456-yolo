import argparse

parser = argparse.ArgumentParser(description='Detect PTW riders and their helmet use on an image.\n'
                                             'Unless otherwise specified, saves results in filename_annotated.png')
parser.add_argument('filename', type=str, help='filename of the image to perform object detection on')
parser.add_argument('-sp', '--savepath', type=str, help='filename to save the results to')
parser.add_argument('-d', '--display', action="store_true", help="whether to display the image immediately")

args = parser.parse_args()
print(args.filename, args.display)
