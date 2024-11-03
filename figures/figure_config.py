import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

figure_config = {
    "title":20,
    "organelle":16,
    "text":10,
    "axis":10,
    "subtitle":12,
    "font":"DejaVu Sans"
}

def get_scalebar():
    scalebar = ScaleBar(0.086, 'um', location='lower right')  # 0.1 micrometers per pixel
    scalebar.box_alpha = 0.5  # Remove background color for the scalebar
    return scalebar

scalebar = get_scalebar()