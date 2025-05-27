# RAPIDA – A semi-automated geospatial analysis tool for rapid crisis response.

At it's core,**rapida** is a python library and command line tool built on top of several
curated geospatial datasets associated with specific geospatial risk's exposure components /variables
capable to perform semi-automated assessments (zonal statistics) over a specific area of interest.

Essentially, **rapida** operates with a special folder or project that contains
       
  - specific metadata in json format
  - vector data layers in [GPKG](https://gdal.org/en/stable/drivers/vector/gpkg.html) format
  - raster data in [COG](https://gdal.org/en/stable/drivers/raster/cog.html) format

A typical **rapida** session consists of following steps:


```mermaid
flowchart LR
    
    i([🟢 Init]) --> a([🔐 Auth])
    as{{🔍 Assess?}}
    %% Switch to top-to-bottom (TB) direction for vertical alignment
    subgraph vertical [ ]
        direction TB
        as <-->|🌥️cloud / 🖧remote| db[(🌍 Data)]
    end
    a --> vertical
    vertical --> u{{📤 upload}}
    u{{📤 upload}} --> p{{🌐 publish}}
    
```

## Features 


- [x] fetching/downloading and curating (h3id) administrative boundaries from [OSM](https://wiki.openstreetmap.org/wiki/Key:admin_level) / [OCHA@COD](https://codgis.itos.uga.edu/arcgis/rest/services) sources
- [x] conducting geospatial assessments over and area of interest containing geospatial polygons
  - [x] buildings
  - [x] deprivation/poverty
  - [x] electrical grid
  - [x] GDP
  - [x] land use
  - [x] population
  - [x] roads
- [x] integration with UNDP Azure cloud storage and [UNDP GeoHub](https://geohub.data.undp.org/)
  - [x] concept of project (create, list, delete, upload, download)
- [x] integration with JupyterHub
  - [x] geospatial visualization
  - [x] notebooks
- [x] rich UI/UX 

## Core software components & libs
+ GDAL www.gdal.org
+ rasterio https://github.com/rasterio/rasterio
+ exactextract https://github.com/isciences/exactextract
+ geopandas https://github.com/geopandas/geopandas
+ rich https://github.com/Textualize/rich/tree/master
+ click https://github.com/pallets/click
+ tensorflow https://www.tensorflow.org/

... and others


## Data sources

> [!IMPORTANT]
> **rapida** operates with public datasets. However, some of these datasets rae currently being hosted in
> [GeoHub](https://geohub.data.undp.org/). This is for two main reasons. First, in case a data source is available from 
> one source only it was moved to Geohub to create a backup. Second, some data sources like population have been also 
> curated and optimized as to facilitate the assessment process.

**rapida** is about conducting geospatial assessments on  geo layers. Various properties/aspects of the layers are 
assessed or evaluated over an area of interest. The variables can be grouped into high level components that correspond
to major areas of evaluation in the context of crises like : population, built environment (buildings, roads), natural
environment (land use/cover, livestock), socio-economical environment (GDP, relative wealth index, deprivation,HDI).


<details>

<summary>
 🧑‍🤝‍🧑 Population
</summary>

Sourced from [WorldPop](https://www.worldpop.org/) project, the components provides country based population data in 
raster format. The constrained (conditioned by buildings), UN adjusted version was selected as it was considered the best
match. As the most recent population dataset was generated for 2020 rapida is forecasting the 2020 population using 
national data from World Bank that is available for several year in the past in respect to 2020. 
From there a coefficient is computed by dividing the target or requested year (ex 2025) to 2020 and the population
statistics computed for 2020 or base year are multiplied with this coefficient.
</details>



 
<details>

<summary style='border-bottom:px solid gray'>
🏙️ Built environment
</summary>

The built environment refers to the human-made surroundings that provide the setting for human activity. This includes 
all physical spaces and infrastructure constructed or significantly modified by people.



1. Buildings
---
This dataset merges Google's V3 Open Buildings, Microsoft's GlobalMLFootprints, and OpenStreetMap building footprints. 
It contains 2,705,459,584 footprints and is divided into 200 partitions. Each footprint is labelled with its respective 
source, either Google, Microsoft, or OpenStreetMap. It can be accessed in cloud-native geospatial formats such as 
GeoParquet, FlatGeobuf and PMTiles.

So far two spatial variables have been defined: the **number of buildings** per polygon to assess and the cumulative
**area of buildings** per polygon to assess.

2. Electrical grid
---

This dataset provides a global-scale, high-resolution predictive map of medium- and low-voltage electricity distribution
networks. Developed using open-source geospatial data, satellite imagery, and machine learning models, it fills critical
data gaps in power infrastructure—particularly in underserved or data-poor regions. The dataset supports energy access 
planning, infrastructure development, and policy-making.


The distribution network layout was estimated using predictive modeling based on spatial features such as 
**population density, proximity to existing infrastructure, land use, and OpenStreetMap data**. 
Models were trained and validated using empirical data from 14 countries, achieving ~75% accuracy. 
The resulting dataset includes both observed and predicted network lines and is suitable for global energy planning and 
accessibility/assessment  studies.




3. Roads
---

The GRIP dataset provides a globally harmonized vector map of road infrastructure, integrating over 21 million kilometers
of roads from a variety of open and commercial sources. Designed to support global-scale environmental and accessibility modeling, 
the dataset offers standardized road classifications and extensive geographic coverage, especially in previously underrepresented regions.

Road data were compiled from national, regional, and global datasets, harmonized into a unified schema, and cleaned for 
spatial consistency. Next, they were classified into five standardized types (e.g., highways, primary, secondary, local, tracks) 
and validated using satellite imagery and supplementary datasets. The dataset is suitable for applications 
targeting land-use modeling, biodiversity impact assessments, and sustainable development planning.


</details>



 
<details>

<summary>
🌾 Landuse
</summary>

While conceptually simple, this layer features several characteristics that make its usage and applications
difficult to interpret. Typically, land use layer is produced using  middle to high resolution 
satellite imagery that is generated by taking snapshots of earth at particular instances of space and time. Next, imagery tiles
that fulfill specific requirements are mosaicked together into a seamless, spatially contiguous dataset.
This brings a large amount of heterogenity into the processing and interpretation of land use layer because various 
pixels that compose it have been acquired at and different times and under different conditions.

In the context of crisis resilience, the timing of satellite image acquisition is critical. Imagery captured before, after, 
or both before and after a specific event is typically used to assess conditions on the ground. In case of land use layer 
this is problematic because neighbour location could be acquired at different instances of time and conditions and bear little resemblance or
positive spatial auto-correlation. 
As a result, **rapida** employes a different approach. Cloud accessible Sentinel 2 L1C imagery, available at global level
is was fed into the [Google dynamic world model](https://github.com/google/dynamicworld) to predict land use in close to real time
for every image selected in a specific time interval with less than 5% cloud coverage. 

The last step that is not yet implemented is to generate the cloud prediction  and use the layer to mask/out filter
cloudy/snowy pixels.

> **Note**
> Predicting land use in close to real time is a computationally demanding task and should be treated with care



</details>


<details>
<summary>
💰 Socio-economic environment
</summary>

The socio-economic environment refers to the social and economic conditions that influence and shape the lives, 
behaviors, and development outcomes of individuals, communities, and societies.

1. Deprivation
---
he Global Gridded Relative Deprivation Index (GRDI), Version 1, provides a detailed picture of relative deprivation and 
poverty worldwide, mapped at a high spatial resolution of approximately 1 km². The index ranges from 0 (least deprived) 
to 100 (most deprived). It is built using a combination of demographic and satellite data, carefully processed to ensure
consistency across different regions. To create GRDI, six key factors were selected to represent different aspects of 
deprivation, such as economic activity, child mortality, and human development. The dataset covers the entire globe, 
integrating the best available data at either a fine-scale grid level or broader administrative boundaries.

GRDI combines six key indicators to assess deprivation levels:

1. Built-up Area Ratio (BUILT): Measures the proportion of land covered by buildings compared to open land. Lower values
    indicate higher deprivation, as rural areas tend to have fewer economic opportunities.¹
2. Child Dependency Ratio (CDR): The number of children (aged 0–14) per 100 working-age adults (15–64). 
   A higher ratio suggests higher deprivation due to greater economic strain on households.²
3. Infant Mortality Rate (IMR): The number of infant deaths (under one year old) per 1,000 live births. 
   A higher IMR indicates poorer health conditions and higher deprivation.
4. Subnational Human Development Index (SHDI): A local version of the Human Development Index (HDI), considering education,
health, and living standards. Lower SHDI scores reflect higher deprivation.³
5. Nighttime Lights (VNL, 2020): Measures light intensity at night, which is often linked to economic activity and infrastructure. 
   Areas with less artificial light tend to have higher deprivation.⁴
6. Nighttime Lights Trend (VNL Slope, 2012–2020): Tracks changes in nighttime lights over time. 
   A decline in brightness suggests worsening deprivation, while an increase indicates economic growth.

The dataset can be used to:
- identify areas with high deprivation to guide poverty reduction efforts
- map socioeconomic inequalities at fine spatial scales
- support policy decisions and resource allocation for targeted interventions


2. Relative Wealth Index
---
The Meta Relative Wealth Index is a high-resolution, machine learning–derived proxy for household wealth, developed by 
Meta’s Data for Good initiative. It estimates relative wealth scores at a 2.4 km grid level across low- and middle-income 
countries by analyzing de-identified Facebook connectivity data and satellite imagery. 
The RWI enables fine-scale economic analysis in data-scarce regions and supports humanitarian, development, and policy interventions.




3. GDP
___
This dataset provides annual, global gridded estimates of Gross Domestic Product (GDP) at 0.1° spatial resolution (~10 km at the equator)
from 2015 to 2100, fully aligned with the five Shared Socioeconomic Pathways (SSPs). 
It offers GDP values in both constant 2015 U.S. dollars and purchasing power parity (PPP), enabling spatially explicit 
long-term economic modeling under diverse socioeconomic scenarios.
The dataset was up-sampled to 1km resolution and reprojected to [EPSG:3857](https://epsg.io/3857) to facilitate web usage.

To construct a spatially explicit and globally consistent GDP dataset aligned with the Shared Socioeconomic Pathways (SSPs), 
the authors began by collecting national-level GDP projections for each of the five SSP scenarios. These projections, 
expressed in both constant 2015 U.S. dollars and purchasing power parity (PPP), were sourced from authoritative institutions 
such as the Organisation for Economic Co-operation and Development (OECD) and the International Institute for 
Applied Systems Analysis (IIASA). 

Next, population data at 0.1° resolution were obtained from the SSP Public Database. 
These gridded population distributions were used as the basis for disaggregating national GDP values. 
The key assumption underpinning the downscaling process was that GDP per capita remains spatially uniform within each 
country for a given year. Based on this, national GDP totals were allocated across grid cells in direct proportion to the
local population count. This method produced a high-resolution, annual global GDP dataset spanning from 2015 to 2100, 
consistent with the spatial and temporal dynamics of each SSP scenario. 
The resulting data are well-suited for use in integrated assessment models, climate impact studies, land-use modeling, and other applications requiring detailed socioeconomic projections.

</details>





## Installation using pip

rapida builds on several open source geospatial packages and has a relative large dependency tree with several native ones
like GDAL and this ads complexity to the installation procedure.

> [!IMPORTANT]
> We recommend installing into a virtual environment as opposed toi install into the default python interpreter/system


___
1. ensure GDAL libs and binary tools are available on your system. ON linux this can be done using:
```commandline
    gdalinfo --version
```
2. create the virtual env suing tools/mean of your choice. Here there are two options
   1. create a virtual end that includes system packages
   ```commandline
   pipenv --python 3 --system-packages
   ```
    2. create a clean virtual end and install GDAL
   ```commandline
   pipenv --python 3
   GDAL_VERSION=$(gdalinfo --version | grep -oP 'GDAL \K[0-9.]+')
   echo $GDLA_VERSION
   pipenv run pip install --no-cache-dir --force-reinstall --no-binary=gdal gdal==$GDAL_VERSION
   pipenv run pip list
    Package Version
    ------- -------
    GDAL    3.8.4
    pip     25.0.1

   ```
3. install rapida from github
   1. with git binaries available
      ```commandline
         pipenv run pip install git+https://github.com/UNDP-Data/rapida.git
      ```
      2. without git binaries
      ```commandline
      pipenv  run pip install https://github.com/UNDP-Data/rapida/archive/refs/heads/main.zip
      ```

4. test the installation
```commandline
pipenv run rapida 
Usage: rapida [OPTIONS] COMMAND [ARGS]...

  UNDP Crisis Bureau Rapida tool.

  This command line tool is designed to assess various geospatial variables
  representing exposure and vulnerability aspects of geospatial risk induced
  by natural hazards.

Options:
  -h, --help  Show this message and exit.

Commands:
  init      initialize RAPIDA tool
  auth      authenticate with UNDP account
  admin     fetch administrative boundaries at various levels from OSM/OCHA
  create    create a RAPIDA project in a new folder
  assess    assess/evaluate a specific geospatial exposure
            components/variables
  list      list RAPIDA projects/folders located in default Azure file share
  download  download a RAPIDA project from Azure file share
  upload    upload a RAPIDA project to Azure file share
  publish   publish RAPIDA project results to Azure and GeoHub
  delete    delete a RAPIDA project from Azure file share

```


## Installation using Docker on Linux/Mac

rapida can be deployed as a docker container and with some shell scripting be used effectively
on a local machine as a command line tool. 

1. ensure docker is installed and working. 
> [!IMPORTANT]
>  Avoid installing docker from snap even if it is convenient. snap docker has some
>  specific restrictions (mount home folder, etc.)
```commandline
    docker --version
```
2. create rapida  docker based launcher

```bash
# 1. Write the script to a temporary file
cat <<'EOF' > /tmp/rapida
#!/usr/bin/bash

if [ -z "$1" ]; then
  command="rapida"
else
  command="$1"
  shift # Remove the command from the arguments if present
fi
VERSION="main"
timestamp=$(date +%s)
docker run --rm -it \
  -u 1000:1000 \
  -e USER=$USER \
  --name rapida$timestamp \
  -m 32GB \
  --cpus 8 \
  -e GDAL_NUM_THREADS=8 \
  -w $PWD \
  -v $PWD:$PWD \
  -v /home:/home \
  -v /data:/data \
  -v /tmp:/tmp \
  ghcr.io/undp-data/rapida:$VERSION \
  $command "$@"
EOF

# 2. Move it to /usr/local/bin
sudo mv /tmp/rapida /usr/local/bin/rapida

# 3. Make it executable
sudo chmod +x /usr/local/bin/rapida


```

**rapida** docker image is based on ghcr.io/osgeo/gdal:ubuntu-small-$VERSION. This is an ubuntu based image with
a non-root ubuntu user (1000:1000) and featuring bash shell. 
The above launcher script ensures that rapida tool installed inside the container can be executed form the local host like
any regular command.

3. test the installation
```commandline
pipenv run rapida 
Usage: rapida [OPTIONS] COMMAND [ARGS]...

  UNDP Crisis Bureau Rapida tool.

  This command line tool is designed to assess various geospatial variables
  representing exposure and vulnerability aspects of geospatial risk induced
  by natural hazards.

Options:
  -h, --help  Show this message and exit.

Commands:
  init      initialize RAPIDA tool
  auth      authenticate with UNDP account
  admin     fetch administrative boundaries at various levels from OSM/OCHA
  create    create a RAPIDA project in a new folder
  assess    assess/evaluate a specific geospatial exposure
            components/variables
  list      list RAPIDA projects/folders located in default Azure file share
  download  download a RAPIDA project from Azure file share
  upload    upload a RAPIDA project to Azure file share
  publish   publish RAPIDA project results to Azure and GeoHub
  delete    delete a RAPIDA project from Azure file share

```

There are several key parameters in the launcher that ensure this process is successful:

- rapida is installed as non-root in a virtual env created by pipenv and made available in the container as a regular script
- the container is run with local logged in linux user using -u 1000:1000
- the USER env variable is **REQUIRED** by rapida to store the auth file
- rapida is a project based tool that requires the command to be invoked from within the folder. The current user folder
 is always passed inside the container as the current working folder using -w flag
- the /home folder is mounted to persist the config files between invoking multiple sessions
- the timestamp ensures multiple projects can be assessed at the same time by ensuring every run creates a container with
  a unique name


## Installation on Windows

## Installation using Docker on Windows

As Docker Engine is not available as a standalone install on Windows for architectural and technical reasons
and as a result we recommend Windows users to use [WSL](https://learn.microsoft.com/en-us/windows/wsl/).

1. [install WSL](https://learn.microsoft.com/en-us/windows/wsl/install) 
2. install Ubuntu inside WSL
   * go to **Start**->type ***Terminal*** and select the application
   * list available wsl images
   ```commandline
   wsl --list --online
    The following is a list of valid distributions that can be installed.
    Install using 'wsl.exe --install <Distro>'.
    
    NAME                            FRIENDLY NAME
    AlmaLinux-8                     AlmaLinux OS 8
    AlmaLinux-9                     AlmaLinux OS 9
    AlmaLinux-Kitten-10             AlmaLinux OS Kitten 10
    Debian                          Debian GNU/Linux
    FedoraLinux-42                  Fedora Linux 42
    SUSE-Linux-Enterprise-15-SP5    SUSE Linux Enterprise 15 SP5
    SUSE-Linux-Enterprise-15-SP6    SUSE Linux Enterprise 15 SP6
    Ubuntu                          Ubuntu
    Ubuntu-24.04                    Ubuntu 24.04 LTS
    archlinux                       Arch Linux
    kali-linux                      Kali Linux Rolling
    openSUSE-Tumbleweed             openSUSE Tumbleweed
    openSUSE-Leap-15.6              openSUSE Leap 15.6
    Ubuntu-18.04                    Ubuntu 18.04 LTS
    Ubuntu-20.04                    Ubuntu 20.04 LTS
    Ubuntu-22.04                    Ubuntu 22.04 LTS
    OracleLinux_7_9                 Oracle Linux 7.9
    OracleLinux_8_7                 Oracle Linux 8.7
    OracleLinux_9_1                 Oracle Linux 9.1
   ```
   * install Ubuntu 22.04 LTS
    ```commandline
    PS C:\Users\user> wsl --install Ubuntu-22.04
    wsl: Using legacy distribution registration. Consider using a tar based distribution instead.
    Installing: Ubuntu 22.04 LTS
    Ubuntu 22.04 LTS has been installed.
    Launching Ubuntu 22.04 LTS...
    Installing, this may take a few minutes...
    Please create a default UNIX user account. The username does not need to match your Windows username.
    For more information visit: https://aka.ms/wslusers
    Enter new UNIX username: rapida
    New password:
    Retype new password:
    passwd: password updated successfully
    Installation successful!
    To run a command as administrator (user "root"), use "sudo <command>".
    See "man sudo_root" for details.
    
    Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.15.167.4-microsoft-standard-WSL2 x86_64)
    
     * Documentation:  https://help.ubuntu.com
       * Management:     https://landscape.canonical.com
       * Support:        https://ubuntu.com/pro
    
     System information as of Fri May 23 09:30:30 UTC 2025
    
      System load:  1.78                Processes:             43
      Usage of /:   0.1% of 1006.85GB   Users logged in:       0
      Memory usage: 18%                 IPv4 address for eth0: 172.20.135.28
      Swap usage:   0%
    
    
    This message is shown once a day. To disable it please create the
    /home/rapida/.hushlogin file.
    ```
3. install docker inside Ubuntu

    * Follow the detailed instruction on how to install docker on ubuntu 22.04
    from [apt repo](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)
    * ensure doker can be run by non-root user
    
        Detailed instructions for ubuntu are available [here](https://docs.docker.com/engine/install/linux-postinstall/)
        ```commandline
        sudo usermod -aG docker $USER
        ```
4. make docker available on Windows host from inside WSL ubuntu

   * This step requires creating a  shell script to launch docker and placing it somewhere in the path
    as **docker.bat** 
   ```commandline
   @echo off
   REM Forward docker CLI commands to Docker inside WSL Ubuntu
    
   REM Pass all arguments from Windows CLI to WSL docker
   wsl docker %*
   ```
   * Close your current terminal and start a new one
   
   ```commandline
    docker
    Usage:  docker [OPTIONS] COMMAND
    
    A self-sufficient runtime for containers
    
    Common Commands:
      run         Create and run a new container from an image
      exec        Execute a command in a running container
      ps          List containers
      build       Build an image from a Dockerfile
      bake        Build from a file
      pull        Download an image from a registry
      push        Upload an image to a registry
      images      List images
      login       Authenticate to a registry
      logout      Log out from a registry
      search      Search Docker Hub for images
      version     Show the Docker version information
      info        Display system-wide information

   ```
   5. create launcher script for rapida

       Rapida tool can be launched now from windows host as a docker image.
       ```commandline
       @echo off
       REM RAPIDA Docker Launcher for Windows
       setlocal ENABLEDELAYEDEXPANSION
    
       REM Set version and timestamp
       set VERSION=main
       for /f %%i in ('powershell -Command "Get-Date -UFormat %%s"') do set TIMESTAMP=%%i
    
       REM Get current directory and convert Windows path to WSL path
       set "CWD=%cd%"
    
       REM Extract drive letter (e.g. C:) and convert to lowercase (e.g. c)
       set "DRIVE_LETTER=%CWD:~0,1%"
       set "DRIVE_LETTER=!DRIVE_LETTER:A=a!"
       set "DRIVE_LETTER=!DRIVE_LETTER:B=b!"
       set "DRIVE_LETTER=!DRIVE_LETTER:C=c!"
       set "DRIVE_LETTER=!DRIVE_LETTER:D=d!"
       set "DRIVE_LETTER=!DRIVE_LETTER:E=e!"
       set "DRIVE_LETTER=!DRIVE_LETTER:F=f!"
       REM Add more drives if needed
    
       REM Remove drive letter and colon from path
       set "DIR_PATH=%CWD:~2%"
       REM Replace backslashes with forward slashes
       set "DIR_PATH=!DIR_PATH:\=/!"
    
       REM Compose WSL-style path for current directory
       set "WSL_PATH=/mnt/!DRIVE_LETTER!!DIR_PATH!"
    
       REM Convert USERPROFILE path to WSL path
       set "UP=%USERPROFILE%"
       set "UP_DRIVE_LETTER=%UP:~0,1%"
       set "UP_DRIVE_LETTER=!UP_DRIVE_LETTER:A=a!"
       set "UP_DRIVE_LETTER=!UP_DRIVE_LETTER:B=b!"
       set "UP_DRIVE_LETTER=!UP_DRIVE_LETTER:C=c!"
       set "UP_DRIVE_LETTER=!UP_DRIVE_LETTER:D=d!"
       set "UP_DRIVE_LETTER=!UP_DRIVE_LETTER:E=e!"
       set "UP_DRIVE_LETTER=!UP_DRIVE_LETTER:F=f!"
       set "UP_PATH=%UP:~2%"
       set "UP_PATH=!UP_PATH:\=/!"
       set "WSL_USERPROFILE=/mnt/!UP_DRIVE_LETTER!!UP_PATH!"
    
       REM Convert TEMP path to WSL path
       set "TMP=%TEMP%"
       set "TMP_DRIVE_LETTER=%TMP:~0,1%"
       set "TMP_DRIVE_LETTER=!TMP_DRIVE_LETTER:A=a!"
       set "TMP_DRIVE_LETTER=!TMP_DRIVE_LETTER:B=b!"
       set "TMP_DRIVE_LETTER=!TMP_DRIVE_LETTER:C=c!"
       set "TMP_DRIVE_LETTER=!TMP_DRIVE_LETTER:D=d!"
       set "TMP_DRIVE_LETTER=!TMP_DRIVE_LETTER:E=e!"
       set "TMP_DRIVE_LETTER=!TMP_DRIVE_LETTER:F=f!"
       set "TMP_PATH=%TMP:~2%"
       set "TMP_PATH=!TMP_PATH:\=/!"
       set "WSL_TEMP=/mnt/!TMP_DRIVE_LETTER!!TMP_PATH!"
    
       REM Debug output (optional)
       echo Using WSL path: !WSL_PATH!
       echo Mounting home: !WSL_USERPROFILE!
       echo Mounting temp: !WSL_TEMP!
    
       REM Run Docker container
    
       docker run --rm -it  -e USER=%USERNAME% --name rapida!TIMESTAMP! -m 2GB --cpus 1 -e GDAL_NUM_THREADS=2 -w "!WSL_PATH!" -v "!WSL_USERPROFILE!":/home ghcr.io/undp-data/rapida:main rapida %*
    
       endlocal
       ``` 
       * save the following code as ***rapida.bat*** and place it somewhere in the %PATH

       * invoke rapida tool script.
   
       This will download the rapida tool docker image and launch it as 
       a container passing all arguments to the tool running  inside
       ``` commandline
           rapida
           Using WSL path: /mnt/c/Users/user
           Mounting home: /mnt/c/Users/user
           Mounting temp: /mnt/c/Users/user/AppData/Local/Temp
           Unable to find image 'ghcr.io/undp-data/rapida:main' locally
           main: Pulling from undp-data/rapida
           802008e7f761: Pull complete
           4f4fb700ef54: Pull complete
           fd687a47324c: Pull complete
           7939e7897b12: Pull complete
           fa651c41f8c7: Pull complete
           38b9cd78de61: Pull complete
           f649ebe50e7d: Pull complete
           92352e5c32d8: Pull complete
           10f330ceb2c0: Pull complete
           dd65c27e736a: Pull complete
           a2c83972aefb: Pull complete
           f273dccc7b00: Pull complete
           fa0c7185a1fb: Pull complete
           e10e0828bfe2: Pull complete
           876f61e413c0: Pull complete
           20236d1d4682: Pull complete
           668bd6a01f96: Pull complete
           73f86a700b9a: Pull complete
           24383bcdcdf2: Pull complete
           1714f3185e4c: Pull complete
           c0ac30670e71: Pull complete
           27e8db7e84ce: Pull complete
           5d33408236ec: Pull complete
           Digest: sha256:a462002cb9e41547c980da6563679ac33e624a87ad3c9ae22b3df031f756224d
           Status: Downloaded newer image for ghcr.io/undp-data/rapida:main
           Usage: rapida [OPTIONS] COMMAND [ARGS]...
        
             UNDP Crisis Bureau Rapida tool.
        
             This command line tool is designed to assess various geospatial variables
             representing exposure and vulnerability aspects of geospatial risk induced
             by natural hazards.
        
           Options:
             -h, --help  Show this message and exit.
        
           Commands:
             init      initialize RAPIDA tool
             auth      authenticate with UNDP account
             admin     fetch administrative boundaries at various levels from OSM/OCHA
             create    create a RAPIDA project in a new folder
             assess    assess/evaluate a specific geospatial exposure
                       components/variables
             list      list RAPIDA projects/folders located in default Azure file share
             download  download a RAPIDA project from Azure file share
             upload    upload a RAPIDA project to Azure file share
             publish   publish RAPIDA project results to Azure and GeoHub
             delete    delete a RAPIDA project from Azure file share
       ```
6. test rapida

> [IMPORTANT]
> Make sure to set the core resources available to the container(CPU, RAM) according to your own system capabilities
   