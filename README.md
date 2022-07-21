# LOD3 buildings (Tested on Ubuntu 18.04lts)
This repository contains the codes for computing geometrical digital twins as LOD3 models for buildings using a structure from motion and semantic segmentation. The methodology hereby implements was presented in the paper [Generating LOD3 building models from structure-from-motion and semantic segmentation" by Pantoja-Rosero et., al. (2022)] (https://doi.org/10.1016/j.autcon.2022.104430)

<p align="center">
  <img src=docs/images/lod3_01.png>
</p>


<p align="center">
  <img src=docs/images/lod3_02.png>
</p>


## How to use it? (Note: tested for ubuntu 18.04lts)

### 1. Clone repository

Clone repository in your local machine. All codes related with method are inside the `src` directory.

### 2. Download data and CNN weights

Example input data can be downloaded from [Dataset for generating LOD3 building models from structure-from-motion and semantic segmentation](https://doi.org/10.5281/zenodo.6651663). This datased contains 5 main folders. SfM, polyfit, weights, data and results. Extract the folders `data/` and `weights/` and place them inside the repository folder

#### 2a. Repository directory

The repository directory should look as:

```
lod3_buildings
└───data
└───docs
└───examples
└───src
└───weights
```

### 3. Environment

Create a conda environment and install python packages. At the terminal in the repository location.

`conda create -n lod3_buildings python=3.7`

`conda activate lod3_buildings`

`pip install -r requirements.txt`

`pip3 install torch torchvision`

### 4. Third party software

The method needs as input Structure from Motion information and LOD2 model that are computed by [Meshroom](https://github.com/alicevision/meshroom) and [Polyfit](https://github.com/LiangliangNan/PolyFit) respectively. Please refeer to the links to know how to use their methodologies.

In addition to create the final 3D LOD3 building models, it is necessary [FreeCAD](https://www.freecadweb.org/downloads.php) python console and its methods. You can either download the appimage and extract their content as `freecad_dev` or download the folder here [freecad_dev](https://drive.google.com/file/d/1LvjPHkhyo_gdBkCyHqN6uEqLqCGaB3vG/view?usp=sharing). Place the folder `freecad_dev` in the repository location. The repository directory should look as:

```
lod3_buildings
└───data
└───docs
└───examples
└───freecad_dev
  └───usr
    └───bin
    └───...
└───src
└───weights
```

### 5. Testing method with pusblished examples

Inside the folder `examples/` we have provide the input scripts that our algorithm needs. Two input scripts are necessary: `..._main_op.py` and `..._main_LOD3.py`. To run for instance the example `p2_LOD3_00_School` simply open the terminal inside the src folder (with the environment activated) and write the next command:

`./LOD3.sh ../examples/p2_LOD3_00_School_main_op.py ../examples/p2_LOD3_00_School_main_LOD3.py`

The algorithm first will create the openings in 3D and then merge them with the LOD2 model. Run the other examples similarly to the previous inline command.

### 6. Creating your own digital twin as LOD3 model

Create a folder `your_LOD3_data_folder` inside the `data\` folder. Inside `your_LOD3_data_folder` create extra folders with the next structure:
```
lod3_buildings
└───data
  └───your_LOD3_data_folder
    └───cameras
    └───images
      └───im1
      └───im2
    └───keypoints
    └───polyfit      
...
```

The methodology requires as input the next:

- cameras.sfm: file containing the poses information (camera position - location, rotation). After running the default `Meshroom` pipeline, this file is output in the folder `MeshroomCache/StructureFromMotion/13231fdsf/`. Copy that file inside the `your_LOD3_data_folder\cameras`
- two registered views images for each facade containing the openings: For each facade, place one image in which the openings are visible in the folder `data/your_LOD3_data_folder/images/im1/` and othere image view in the folder `data/your_LOD3_data_folder/images/im2/`.
- polyfit.obj: use `Polyfit` pipeline either with the sparse or dense point cloud produced by the `Meshroom` pipeline. Note that it might be necessary to pre-process the point clouds deleting noisy points before running `Polyfit`. Save the output file as polyfit.obj or polyfit_dense.obj and place it in the folder `data/your_LOD3_data_folder/polyfit/`

Check the files of the data examples provided if neccessary to create the inpute data.

Finally create the two input scripts (`your_LOD3_main_op.py` and `your_LOD3_main_LOD3.py`) following the contents the given examples. Open the terminal inside the src folder (with the environment activated) and write the next command:

`./LOD3.sh ../examples/your_LOD3_main_op.py ../examples/your_LOD3_main_LOD3.py`


### 7. Results

The results will be saved inside `results` folder. Images of the pipeline stages are saved together with .obj files for the openings in 3D and the LOD3 model.

#### 7.a Final repository directory

The repository directory after runing the medothology looks as:

```
lod3_buildings
└───data
└───docs
└───examples
└───freecad_dev
└───results
└───src
└───weights
```

### 8. Citation

We kindly ask you to cite us if you use this project, dataset or article as reference.

Paper:
```
@article{Pantoja-Rosero2022b,
title = {Generating LOD3 building models from structure-from-motion and semantic segmentation},
journal = {Automation in Construction},
volume = {141},
pages = {104430},
year = {2022},
issn = {0926-5805},
doi = {https://doi.org/10.1016/j.autcon.2022.104430},
url = {https://www.sciencedirect.com/science/article/pii/S092658052200303X},
author = {B.G. Pantoja-Rosero and R. Achanta and M. Kozinski and P. Fua and F. Perez-Cruz and K. Beyer},
}
```
Dataset:
```
@dataset{Pantoja-Rosero2022b-ds,
  author       = {Pantoja-Rosero, Bryan German and
                  Achanta, Radhakrishna and
                  Kozinski, Mateusz and
                  Fua, Pascal and
                  Perez-Cruz, Fernando and
                  Beyer, Katrin},
  title        = {{Dataset for generating LOD3 building models from 
                   structure-from-motion and semantic segmentation}},
  month        = jun,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {v.0.0},
  doi          = {10.5281/zenodo.6651663},
  url          = {https://doi.org/10.5281/zenodo.6651663}
}
```
