# %% imports
from image_utils import readImage, rescaleImage, writeImage

# %% run main script
photos = [
    "photo001",
    "photo002",
    "photo003",
    "photo004",
    "photo005",
    "photo006",
    "photo007",
    "photo008",
    "photo009",
    "photo010",
    "photo011",
]
scaleRatio = 0.1

for photo in photos:
    originalImage = readImage(f"./input/{photo}.jpg")
    rescaledImage = rescaleImage(originalImage, scaleRatio)
    writeImage(f"./temp/{photo}.jpg", rescaledImage)
