# pixi based deployment

Rapida provides a suite of tools for real-time crisis assessment, relying on a vast ecosystem of software libraries. 
Because many of these dependencies are written in C/C++ for performance and require complex, dynamic linking to 
system-level libraries, installation can be notoriously difficult. 
To eliminate this barrier, Rapida leverages pixi — a **fast, modern, and highly reproducible** package manager 
that guarantees a seamless setup across all platforms.

Pixi defaults to the biggest Conda package repository, conda-forge, which contains over 30,000 packages.
## Install pixi

Refer to [pixi installation](https://pixi.prefix.dev/latest/#installation) docs

## Download

**Linux & macOS**
```bash
curl -O https://raw.githubusercontent.com/UNDP-Data/rapida/refs/heads/main/deploy/pixi.toml
```
**Windows**
```shell
curl.exe -O https://raw.githubusercontent.com/UNDP-Data/rapida/refs/heads/main/deploy/pixi.toml
```

## .env  file

```shell
# for uploading to azure
TENANT_ID=
CLIENT_ID=
# for downloading NTL data from Black Marble
EARTHDATA_TOKEN=
# for predicting precisely VIIRS orbits
SPACETRACK_USER=
SPACETRACK_PASSWORD=

# override road type default seed for connectivity analysis

#"motorway": 105,
#"trunk": 90,
#"primary": 75,
#"secondary": 60,
#"tertiary": 50,
#"unclassified": 40,
#"residential": 35,
#"service": 25

MJOLNIR_SECONDARY_SPEED=40
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