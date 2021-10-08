# CODEX celltype trainer orientation

Welcome to this very, very pre-alpha super in-development multiplex image data viewer + annotation + classification interface. This page is meant to provide an orientation to the interface and main functions of the viewer + trainer. For detailed questions or to report things not working it would be helpful to open a new issue.

1. [Loading data](#loading-data)
2. [Using the viewer](#using-the-viewer)
3. [Annotations](#Annotations)
4. [Train and deploy a classifier](#train-and-deploy-a-classifier)
5. [Config files](#config-files)

General app layout

![welcome](assets/welcome_screen.png)

-----

## Loading data

Available datasets are populated in a dropdown menu on top of the data loading pane. You should select the region and in the next two dropdown menus select config files.

Click `Initialize`.

No data is loaded yet.  
Initialize instructs the viewer to read the configs, and internally we figure out where the source images for the selected dataset reside. 
To the right, the image controls will populate widgets to configure the visibility, color, and low/high saturation points for each channel in the color config file.
The celltype config file is used to quickly select combinations of channels. Each `celltype` lists a set of active channels that can be used to distinguish those cells in the image. On initialization, the first-listed celltype is automatically chosen. 

With the dataset initialized, we can move on to loading and viewing an image.

-----

## Using the viewer

On top of the image controls panel, there is a toggle between `Overview` and `1x` mode for image viewing. Keep this on `Overview`.

Also in the image controls panel, there is a numerical input field that is by default set to `0.2`. This field controls the downsampling applied to the Overview image. The suggestion is to use a low resolution (value between 0.05 - 0.2) to set color saturation values, then switch this to higher resolution later. 

![defaults](assets/default_settings.png)

Leaving all settings default, click `Next image`. 

If you have the terminal process available, you will be able to monitor the process as it loads images from disk, applies the requested downsampling, then assembles the active channels into a merged RGB image. 

The image display area will update, which may take time depending on network latency and the volume of data to be pushed.

![zoom](assets/wheel_zoom.png)

Click the scroll wheel icon to enable zooming. Click + drag panning should be enabled by default.

### Setting colors

![colors](assets/color_controls.png)

- Click the channel name to toggle visibility
- Each channel's color can be set independently by clicking on the color widget
- For each channel, values below the `low cutoff` are set to 0, i.e. made invisible
- For each channel, values above the `high saturation` are clipped. 

Images are colorized so that the intensity range between low cutoff and high saturation map to a point on a uniform color ramp from black (`values = low cutoff`) and the full intensity color (`values = high saturation`). 

The displayed image is a linear combination of the colorized version of each channel after the cutoff/saturation is applied. So, if images appear too 'white' after blending, try to increase the high saturation values for some bright channels. Alternatively, choose a "darker" color. A frequent culprit for 'white' images is the DAPI (or nuclear) channel, which is set to a light-gray by default, and therefore contributes intensity to all 3 RGB channels. Increasing the DAPI channel saturation value (making it appear dimmer) often frees up intensity range for the other, more interesting, channels to shine.

-----

## Annotations

### Loading annotations

### Making new annotations

### Saving annotations

-----

## Train and deploy a classifier


------

### Config files