# pixi based deployment

Rapida provides a suite of tools for real-time crisis assessment, relying on a vast ecosystem of software libraries. 
Because many of these dependencies are written in C/C++ for performance and require complex, dynamic linking to 
system-level libraries, installation can be notoriously difficult. 
To eliminate this barrier, Rapida leverages pixi — a **fast, modern, and highly reproducible** package manager 
that guarantees a seamless setup across all platforms.

Pixi defaults to the biggest Conda package repository, conda-forge, which contains over 30,000 packages.

## Windows

### 1. install git
```shell
git winget install --id Git.Git -e --source winget

```

### 2. install pixi
```shell
git winget install --id Git.Git -e --source winget

```

### 3. download pixi cfg 

```shell
curl.exe -O https://raw.githubusercontent.com/UNDP-Data/rapida/refs/heads/main/deploy/pixi.toml
```


### 4. install 
```shell
pixi install

```

### 5. setup playwright 
```shell
pixi  run setup

```


### 6. run in local folder 
```shell
PS C:\Users\rapida\rapida> pixi run rapida
✨ Pixi task (rapida): dotenv -e .env rapida
Usage: rapida [OPTIONS] COMMAND [ARGS]...

  UNDP Crisis Bureau Rapida tool.

  This command line tool is designed to assess various geospatial variables
  representing exposure and vulnerability aspects of geospatial risk induced
  by natural hazards.

Options:
  --help  Show this message and exit.

Commands:
  init          initialize RAPIDA tool
  auth          authenticate with UNDP account
  admin         fetch administrative boundaries at various levels from
                OSM/OCHA
  create        create a RAPIDA project in a new folder
  assess        assess/evaluate a specific geospatial exposure
                components/variables
  list-project  list RAPIDA projects/folders located in default Azure file
                share
  download      download a RAPIDA project from Azure file share
  upload        upload a RAPIDA project to Azure file share
  publish       publish RAPIDA project results to Azure and GeoHub
  delete        delete a RAPIDA project from Azure file share
  addh3id       add h3id to a vector dataset
  population    Population data management commands.
  ntl           Nighttime Lights VIIRS data and impact detection
  connectivity  run connectivity analysis

```


### 7. make rapida accessible globally 
```shell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/UNDP-Data/rapida/refs/heads/main/deploy/install.ps1 | iex"

```




## Linux & Mac

### 1. install git

See [Git installation instructions](https://git-scm.com/install/)

### 2. install pixi
```shell
curl -fsSL https://pixi.sh/install.sh | sh
```
```shell
# or 
wget -qO- https://pixi.sh/install.sh | sh

```

### 3. Download pixi config

```bash
curl -O https://raw.githubusercontent.com/UNDP-Data/rapida/refs/heads/main/deploy/pixi.toml
```
### 4. install 
```shell
pixi install

```

### 5. setup playwright 
```shell
pixi  run setup

```



## Environmental variables

There are two types of environmental variables: mandatory and optional.
The **mandatory** ones contain information that is required to operate specific parts of rapida like
downloading imagery from earthdata through a token or authenticating to [space-track.org](www.space-track.org).
Rapida throws and error if these variables are not defined and needed.

```shell
# for uploading to azure
TENANT_ID=
CLIENT_ID=
# for downloading NTL data from Black Marble
EARTHDATA_TOKEN=
# for predicting precisely VIIRS orbits
SPACETRACK_USER=
SPACETRACK_PASSWORD=
```

The **optional** environmental variables do not generate errors if not defined. Rapida lets the user know whenever these
variables arte detected and detected and override/change default behaviour with a custom one.

```shell
# override road type default seed for connectivity analysis
#Default values

#"motorway": 105,
#"trunk": 90,
#"primary": 75,
#"secondary": 60,
#"tertiary": 50,
#"unclassified": 40,
#"residential": 35,
#"service": 25
# variables that change default values

#MJOLNIR_MOTORWAY_SPEED=60
#MJOLNIR_TRUNK_SPEED=50
#MJOLNIR_PRIMARY_SPEED=35
#MJOLNIR_SECONDARY_SPEED=25
#MJOLNIR_UNCLASSIFIED_SPEED=15
#MJOLNIR_RESIDENTIAL_SPEED=12
#MJOLNIR_SERVICE_SPEED=8
#CONNECTIVITY_OSM_SOURCE=MOVISDA # MOVISDA, GEOFABRIK

```




## run NTL
```shell
pixi run rapida ntl detect --help
✨ Pixi task (rapida): dotenv -e .env rapida ntl detect --help
Usage: rapida ntl detect [OPTIONS]

Options:
  -b, --bbox BBOX                 Bounding box xmin/west, ymin/south,
                                  xmax/east, ymax/north  [required]
  --date [%Y-%m-%d]               The human experience of a specific night,
                                  local time zone matched to the center of
                                  bbox  [required]
  --dst-dir DIRECTORY             Destination directory to save the downloaded
                                  the images.  [default: /tmp]
  -d [noaa_outage|nasa_nrt_outage|nasa_outage]
                                  One or more of the RAPIDA NTL deliverables.
                                  [required]
  --popvar TEXT                   One or more RAPIDA population variable to
                                  compute zonal stats for outages
  -ot, --percentage_drop INTEGER  Specify the outage drop threshold that wil
                                  determine the spatial structure of an outage
                                  event,
  -cm, --cmask                    Enable strict Cloud Masking (ignores pixels
                                  with NASA quality flags of 3). Disable this
                                  flag during major storm events to prevent
                                  atmospheric noise from erroneously masking
                                  out legitimate blackout signals.
  --display                       Show a graphic visualization of the outage
                                  analysis.Useful to inspect the input imagery
                                  and debug/understand the outage results
  --debug                         Enable debug logging.
  --help                          Show this message and exit.



pixi run rapida ntl search noaa -b -72.3,7.5,-64.16,13.72 --date 2026-07-10 -cm
✨ Pixi task (rapida): dotenv -e .env rapida ntl search noaa -b -72.3,7.5,-64.16,13.72 --date 2026-07-10 -cm
[07/17/26 01:32:59] INFO     Skipping DescendingPass: SNPP-202607100504 because of low elevation angle 13.664044                                                                 search.py:396
[07/17/26 01:33:07] INFO     Skipping DescendingPass: N21-202607100429 because of low elevation angle 4.678326                                                                   search.py:396
[07/17/26 01:33:09] INFO     Skipping DescendingPass: N21-202607100749 because of low elevation angle 2.797944                                                                   search.py:396
                                      VIIRS satellites granules for the night of  2026-07-10 covering (-72.3, 7.5, -64.16, 13.72)                                       
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Position ┃ Satellite ┃ Timestamp (UTC) ┃ Bbox offset from SSP (km) ┃ Elevation above bbox (degrees) ┃ Cloud coverage in bbox (%) ┃ Score (%) ┃ BBOX intersection (%) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│    1     │    N21    │  202607100609   │            115            │             80.81              │             89             │    42     │          81           │
│    2     │   SNPP    │  202607100644   │            798            │             39.82              │             88             │    24     │          75           │
│    3     │    N20    │  202607100523   │           1191            │             26.91              │             93             │    15     │          43           │
│    4     │    N20    │  202607100703   │           1443            │             20.95              │             96             │    11     │          35           │
└──────────┴───────────┴─────────────────┴───────────────────────────┴────────────────────────────────┴────────────────────────────┴───────────┴───────────────────────┘


```