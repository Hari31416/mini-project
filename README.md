# Impact Dynamics of Liquid Drop
## The Structure of Files and Directories
The root directory is divided into three sub-directories:
1. **codes** containing (almost) all the code related files.
2. **data** containing all the *input* as well as *output* data
3. **presentations** containing codes and other assets for the presentation and the article 

### The `code` Directory
The root of the `code` directory has some jupyter notebooks which we wrote while working on implementing methods to track the trajectory of the drop. Apart from these, the directory has two folders
1. **extraction** has a single notebooks for each video sequence and describes everything we have done to extract the useful information from the video. The name of the file corresponds to the title of the tiff file.
2. **html** has HTML version of some of the notebooks which are hosted online and can be accessed using the [url](https://hari31416.github.io/mini-project/).

### The `data` Directory
At the root of the directory are  notebooks containing some *helper code*. The folder has following sub-directories:
1. **animation** contains the animation of the drop moving through the thin film with the center of the drop detected by our method. The name of the file corresponds to the title of the tiff file.
2. **extracted_images** has the images extracted from the tiff file. The name of the folder corresponds to the title of the tiff file.
3. **final_df** contains the final CSV file which has every extracted information
4. **Images1** has images from the first video
5. **plots** has some graphs of the fitted curves. The name of the folder corresponds to the title of the tiff file. Each folder has plot for X and Y with time as well as X with Y using the parametric curve fitting and the parabola fitting.
6. **results** contains everything we have extracted. The folder is organized in many categories with the folder name corresponding to the title of the tiff file. The files inside each subfolders has the same structure. Also, at the root of the directory, we have some CSV files containing the *overall* information from all the video. See the [Info About Columns](data/results/info_about_columns.md) file to get an idea of what the various columns of the dataframes means.
7. **tiff_files** has the tiff files for each videos organized in two directories
    - Varying Height
    - Varying Angle

    >The folder is large hence been added to the `.gitignore` file and won't be availbale on Github.

### The `presentations` Directory 
This directory contains two sub directories:
1. **first** containing files related to the presentations
2. **second** containing files related to the article