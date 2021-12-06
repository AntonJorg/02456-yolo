# 02456-yolo 😎
![Spicy example](example.gif)

Repository for a PyTorch implementation of the YOLO model for the course 02456 Deep Learning at the Technical University of Denmark.

## Usage 🚀 
To annotate a single image, run the `detect.py` script as follows:
```
python detect.py filename.extension
```
This will save the annotated image as `filename_annotated.png`, in the same location as `filename.extension`.

To annotate an entire folder of images and compile a GIF with the results:
```
python detect.py --dir directory --savedir output_directory --gif
```
This will save all the individual annotated images and the GIF in `output_directory`.

For more help, see
```
python detect.py -h
```

## Architecture 🧞


## Data 💻

HELMET dataset: https://osf.io/4pwj8/


## Authors 🤓

- Alma Fazlagic
- Anton Thestrup Jørgensen
- Rasmus Wael Lind
- Rune Henrik Verder Sehested
