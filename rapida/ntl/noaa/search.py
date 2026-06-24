"""
Search for VIIRS satellites passes using pyrobital and TLE
"""
import json
import os.path
from pyorbital.orbital import Orbital
from datetime import datetime, timedelta, date, time as dtime
import math
from dataclasses import dataclass, asdict
from rich.progress import Progress
import logging
from typing import Iterable, Optional
import obstore
from rapida.ntl.noaa.io import (
find_ntl, public_url, parse_noaa_timestamp, VIIRS_STORES, SOURCE_NAMES, PRODUCTS
)
import asyncio
from spacetrack import SpaceTrackClient
from rapida.ntl.noaa.cmask import cloud_coverage_batch
from rapida.ntl.noaa.const import  PRODUCTS_RE
from rapida.ntl import cache
import numpy as np
logger = logging.getLogger(__name__)


TLE_URL = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=weather'

@dataclass
class DescendingPass:
    rise_time:datetime
    fall_time:datetime
    max_elev_time:datetime
    target_date:date
    sat:str

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f'{self.__class__.__name__}: {self.id}'
    @property
    def id(self):
        return f'{self.sat}-{self.max_elev_time:%Y%m%d%H%M}'

@dataclass
class Granule:
    sat:str
    start_time:datetime
    offset:int
    elevation:float
    cloud_cover = None
    pint:float = None
    @property
    def id(self):
        return f"{self.start_time:%Y%m%d%H%M%S}{self.start_time.microsecond // 100000}" #

    @property
    def timestamp(self):
        return f'{self.start_time:%Y%m%d%H%M}'

    @property
    def sat_rank(self):
        # Pure geometry score based strictly on elevation.
        # 90 degrees (zenith) = 100 points. 0 degrees (horizon) = 0 points.
        return int((self.elevation / 90.0) * 100)

    @property
    def rank(self):
        # The final score, heavily weighting clear skies over pure geometry
        if str(self.cloud_cover).isnumeric():
            clear_sky_score = 100 - self.cloud_cover
            # 70% Weather, 30% Geometry
            return int((self.sat_rank * 0.4) + (clear_sky_score * 0.6))

        return int(self.sat_rank)

    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        ddict = dict(satellite=self.sat, timestamp=self.timestamp,offset_km=self.offset,elevation=self.elevation, cloud_coverage=self.cloud_cover, rank=self.rank, bbox_perc_intersection=self.pint)
        return json.dumps(ddict, separators=(',', ':'),)
    def __repr__(self):
        return f'{self.sat} granule {self.id}  with sat rank {self.sat_rank:0d} and offset {self.offset} km from SSP featuring elevation of {self.elevation:.0f} degrees '




def get_satellite_phase(timestamp_str:str=None, sat_name=None, tle_file="weather.tle"):
    """
    Computes the Phase Offset for a satellite based on a known 'Golden' timestamp extracted from
    the first file of a given day.
    The Phase Offset is a mission-specific constant required to align the internal pyorbital
    propagator with the NOAA ground system’s granule-segmentation logic
    Why it is required:
        The VIIRS instrument generates data in 85.4-second increments (granules).
        However, these granules do not necessarily start at 00:00:00 relative to a TLE Epoch.
        The Phase Offset acts as the "Temporal Anchor," shifting the theoretical pulse train to match the physical
        filenames found in the S3 bucket.

        Calibration Procedure:

            Identify a high-quality (Near-Nadir) file in the S3 bucket.

            Extract the timestamp from the filename (e.g., d20260411_t2246330).

            Calculate the difference between this timestamp and the current TLE Epoch.

            Apply the Modulo 85.4 operation to extract the offset.

    timestamp_str: Format 'dYYYYMMDD_tHHMMSSs' (e.g., 'd20260412_t0000337')
    sat_name: Name of the satellite (e.g., 'SUOMI NPP')

    phase = get_satellite_phase(timestamp_str='d20260412_t0000347', sat_name='NOAA-21')
    where d20260412_t0000347 is SVDNB_j02_d20260412_t0000347_e0001576_b17716_c20260412002520026000_oebc_ops.h5
    FIRST file in a day

    these phases are necewssary for the func to work. Apparently tey need regular yearly calibration??

    """
    # 1. Parse the string into a high-precision datetime
    # Format: d20260412_t0000337
    date_part, time_part = timestamp_str.split('_')

    # Remove 'd' and 't' prefixes
    ds, ts = date_part[1:], time_part[1:]

    # Parse YYYYMMDDHHMMSS
    # We strip the last digit (decisecond) for the initial parse
    t_file = datetime.strptime(ds + ts[:-1], "%Y%m%d%H%M%S")

    # Add the decisecond as microseconds (1 decisecond = 100,000us)
    decisecond = int(ts[-1])
    t_file = t_file.replace(microsecond=decisecond * 100000)
    # 2. Load Orbital data to get the Epoch
    orb = Orbital(sat_name, tle_file=tle_file)
    t_epoch = orb.orbit_elements.epoch

    # 3. The "No-Beta" Pulse Math
    # Hardware pulse duration (1025 / 12 seconds)
    GRANULE_DUR = 85.416666667

    # Calculate the remainder (Phase) relative to the TLE Epoch
    delta = (t_file - t_epoch).total_seconds()
    phase = delta % GRANULE_DUR

    return phase

async def granules2files(granules: list[Granule]=None, satellite: str = None,
                         bbox: Iterable[float] = None, product: str = 'CM',
                         progress=None):
    results = {}

    # Safety check: if there are no granules, just return early
    if not granules:
        return results

    progress_task = None
    if progress:
        progress_task = progress.add_task(
            description=f'[cyan]Evaluating {len(granules)} granule(s) for satellite {satellite}',
            total=len(granules)
        )

    async def track_task(agranule):
        try:
            return await find_ntl(
                satellite=satellite,
                dt=agranule.start_time,
                products=[product],
                bbox=bbox,
            )
        finally:
            if progress and progress_task is not None:
                progress.update(progress_task, advance=1)

    try:
        async with asyncio.TaskGroup() as tg:
            task_map = {tg.create_task(track_task(g)): g for g in granules}

        results = {g: t.result() for t, g in task_map.items()}

        if progress and progress_task is not None:
            progress.update(progress_task, description = f'Selected {len(results)} granule(s) for satellite {satellite}')
            progress.remove_task(progress_task)
        return results

    except ExceptionGroup as eg:
        for e in eg.exceptions:
            logger.error(f"❌ Sub-task failed: {e}")
        return results


class VIIRSNavigator:
    """
    A physics-based navigator for VIIRS (S-NPP, NOAA-20/21).
    Synchronizes the 85.4s instrument heartbeat to the TLE Epoch.
    """

    # VIIRS Hardware Constant (1025 packets / 12)
    GRANULE_DUR = 1025/12.
    # THE OFFLINE MASTER SEEDS (Locked in April 2026)
    # Drift is 'Seconds shifted per 24 hours'
    #
    # the drifting was comuted by analyiz the timestamp of the first image produced by each satellite
    # ex for SNPP using rclone
    # for i in $(seq 0 30); do T_DATE=$(date -d "2026-04-15 - $i days" +%Y/%m/%d); echo -n "$T_DATE | "; rclone lsf --s3-provider AWS --s3-region us-east-1 --s3-no-check-bucket ":s3:noaa-nesdis-snpp-pds/VIIRS-DNB-SDR/$T_DATE/" --include "*t00*.h5" -q | sort | head -n 1 | grep -o 't[0-9]\{7\}' | sed 's/t//'; done

    SATELLITES = {'SNPP':'SUOMI NPP', 'N20':'NOAA 20', 'N21':'NOAA 21'}


    MIN_ELEVATION_ANGLE = 20

    def __init__(self, satellite=None):
        self.satellite = satellite



    def fetch_cached_tle(self, target_date: date=None, cache=cache ) -> str:
        """
        Fetches the TLE for a specific date, checking the shelve cache first.
        If missing, hits Space-Track, caches it, and returns the formatted string.
        """
        # Deterministic cache key
        key = f"TLE_{self.satellite}_{target_date.strftime('%Y%m%d')}"
        tle_data = cache.fetch(key=key)
        if not tle_data:
            user = os.getenv('SPACETRACK_USER')
            if user is None:
                raise EnvironmentError("SPACETRACK_USER is missing from environment variables.")

            passwd = os.getenv('SPACETRACK_PASSWORD')
            if passwd is None:
                raise EnvironmentError("SPACETRACK_PASSWORD is missing from environment variables.")
            # 2. Network Fetch (Space-Track)
            st = SpaceTrackClient(
                identity=user,
                password=passwd
            )

            cat_ids = {"SNPP": "37849", "N20": "43013", "N21": "54234"}
            start_date = (target_date - timedelta(days=3)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')

            tle_data = st.gp_history(
                norad_cat_id=cat_ids[self.satellite],
                epoch=f"{start_date}--{end_date}",
                orderby='EPOCH desc',
                limit=1,
                format='tle'
            )

            if not tle_data:
                raise ValueError(f"No TLE found for {self.satellite} on {target_date}")
            cache.store(key=key, value=tle_data)


        return tle_data



    def get_orbital(self, target_date:date=None):
        tle_text = self.fetch_cached_tle(target_date=target_date)

        l1, l2 = (l.strip() for l in tle_text.strip().splitlines() if l.strip())

        return Orbital(satellite=self.satellite, line1=l1, line2=l2)



    def get_phase_for_date(self, target_date:datetime.date=None):
        orb = self.get_orbital(target_date=target_date)
        t_epoch = orb.orbit_elements.epoch
        # FIXED MATH: Force pyorbital's numpy.datetime64 into a standard Python datetime
        if isinstance(t_epoch, np.datetime64):
            # .item() cleanly converts a scalar numpy datetime back to standard python datetime
            t_epoch = t_epoch.astype('datetime64[us]').item()
        stores = VIIRS_STORES[self.satellite]
        first_file_path = None
        for source in SOURCE_NAMES:

                store = stores[source]
                product = PRODUCTS['CM']
                date_path = target_date.strftime('/%Y/%m/%d/')
                prefix = f"{product}{date_path}"
                stream = obstore.list(store, prefix=prefix)


                for chunk in stream:
                    for meta in chunk:
                        if meta["path"].endswith('.nc'):
                            first_file_path = meta["path"]
                            break
                    if first_file_path:
                        break
                if not first_file_path: # move to another source

                    continue
        if first_file_path is None:

            logger.info(f'No VIIRS operational data was found for {self.satellite} {target_date.date()} in {" or ".join(SOURCE_NAMES)}. '
                        f'Please check manually at  https://noaa-nesdis-{self.satellite.lower()}-pds.s3.amazonaws.com/index.html#VIIRS-JRR-CloudMask/  ')
            return
        _, filename = os.path.split(first_file_path)
        m = PRODUCTS_RE['CM'].match(filename)
        if m:
            start_time = parse_noaa_timestamp(m.groupdict()['start'])
            # 5. Calculate physical phase (Delta % Granule Duration)
            delta = (start_time - t_epoch).total_seconds()
            return delta % self.GRANULE_DUR



    def decompose_bbox(self, bbox:Iterable[float]=None):

        minlon, minlat, maxlon, maxlat = bbox


        midlon = (minlon + maxlon) *.5

        # Latitudes: Top for the trigger, Center for the math

        midlat = (minlat + maxlat) *.5

        return midlon, midlat, maxlat


    def pass2granule(self, p:DescendingPass=None, midlon:float=None, midlat:float=None, elevation:float=None ):
        phase = self.get_phase_for_date(target_date=p.target_date)
        if phase is None:
            return
        orb = self.get_orbital(p.target_date)
        sat_lon, _, _ = orb.get_lonlatalt(p.max_elev_time)
        deg_offset = abs(midlon - sat_lon)
        # Physical distance in km at this latitude
        offset_km = int(deg_offset * 111.32 * math.cos(math.radians(midlat)))
        # Anchor to Midnight UTC of the target day
        t_midnight = datetime.combine(p.target_date.date(), dtime(0, 0, 0))
        delta_seconds = (p.max_elev_time - t_midnight).total_seconds()

        # 2. Pulse-Sync Math
        pulse_index = math.floor((delta_seconds - phase) / self.GRANULE_DUR)
        start_time = t_midnight + timedelta(seconds=(pulse_index * self.GRANULE_DUR) + phase)


        return Granule(sat=self.satellite,start_time=start_time,offset=offset_km, elevation=elevation)

    def night_passes(self, bbox:Iterable[float]=None, nominal_date:date=None):
        orb = self.get_orbital(target_date=nominal_date)
        midlon, midlat, northlat = self.decompose_bbox(bbox=bbox)

        # 1. NIGHT DURATION (Use Mid-Lat for 'Average' Night)
        doy = nominal_date.timetuple().tm_yday
        declination = 0.409 * math.sin(2 * math.pi * (doy - 80) / 365)
        lat_rad = math.radians(midlat)
        cos_h = -math.tan(lat_rad) * math.tan(declination)
        night_hrs = int(round(24 - (2 * math.degrees(math.acos(max(-1.0, min(1.0, cos_h)))) / 15)))
        search_duration_hrs = max(10, night_hrs)
        # 2. THE ANCHOR (01:30 AM Local -> UTC)

        utc_anchor = datetime.combine(nominal_date, dtime(1, 30)) - timedelta(hours=midlon / 15.0)

        # 3. THE TRIGGER (Use North-Lat to find when the satellite ENTERS the box)
        search_start = utc_anchor - timedelta(hours=search_duration_hrs / 2)
        night_passes = orb.get_next_passes(search_start, search_duration_hrs, midlon, midlat, 0) # northlat???
        logger.debug(f'{self.satellite} passes {len(night_passes)} time(s) over {list(bbox)} on the night of {nominal_date:%y-%m-%d}')
        passes = []
        for _pass_ in night_passes:
            rise_time, fall_time, max_elev_time = _pass_
            # Direction Check
            pos_start = orb.get_lonlatalt(rise_time)
            pos_end = orb.get_lonlatalt(fall_time)

            if not pos_end[1] < pos_start[1]:  # Descending
                logger.debug(f'Skipping ascending pass {_pass_}')
                continue
            p = DescendingPass(sat=self.satellite, rise_time=rise_time, fall_time=fall_time,
                               max_elev_time=max_elev_time, target_date=nominal_date)

            passes.append(p)
        return passes


    async def night_granules_async(self, bbox:Iterable[float]=None, nominal_date:date=None, cmask=False, progress=None):

        midlon, midlat, northlat = self.decompose_bbox(bbox=bbox)
        passes = self.night_passes(nominal_date=nominal_date, bbox=bbox)
        orb = self.get_orbital(target_date=nominal_date)
        selected_granules = {}
        granules = []

        for p in passes:
            look = orb.get_observer_look(p.max_elev_time, midlon, midlat, 0)
            elevation = look[1]
            if elevation < self.MIN_ELEVATION_ANGLE:
                logger.info(f'Skipping {p} because of low elevation angle {elevation:0f}')
                continue

            granule = self.pass2granule(p=p,midlon=midlon, midlat=midlat, elevation=elevation,)
            if granule:
                granules.append(granule) # return noe in case no data is available with hyper-scalers
            else:
                logger.info(f'Could not convert pass {p} to granule. Skipping.')



        geom_granules = await granules2files(
            granules=granules,satellite=self.satellite, bbox=bbox, progress=progress
        )


        for current_granule, found in geom_granules.items():

            if not found:
                continue
            # Safely get the first source/entry
            (source, entries), = found.items()

            file_path, file_size, p = entries[0]
            current_granule.pint = p
            _, file_name = os.path.split(file_path)
            if f's{current_granule.timestamp}' not in file_name:
                m = PRODUCTS_RE['CM'].match(file_name)
                if m:
                    start_time = parse_noaa_timestamp(m.groupdict()['start'])
                    old_timestamp = current_granule.timestamp
                    current_granule.start_time = start_time
                    logger.debug(f'Replacing granule {old_timestamp} with {current_granule.timestamp}')

            if cmask:
                # Use the unique URL as the key (Always unique)
                url = public_url(file_path=file_path, satellite=self.satellite, source=source)
                selected_granules[url] = current_granule
            else:
                # Use the unique file_path as the key to prevent SNPP/N20/N21 overwrites
                selected_granules[file_path] = current_granule


        return selected_granules



async def async_search_granules(
        satellites:Optional[Iterable[str]]=None, nominal_date:date=None, bbox:Iterable[float] = None,
        cmask=False, progress=None, push_to_cache:bool=False,
    ):
    """

    :param satellites:
    :param nominal_date:
    :param bbox:
    :param cmask:
    :param progress:
    :return:
    """

    satellite_names = list(VIIRSNavigator.SATELLITES.keys())
    assert isinstance(nominal_date, date), f'invalid target date {nominal_date}'
    satellites = satellites or satellite_names
    selected_granules = []
    found_granules = {}

    tasks = []
    progress_task = None
    try:
        async with asyncio.TaskGroup() as tg:
            for sat in satellites:
                nav = VIIRSNavigator(satellite=sat)
                tasks.append(tg.create_task(
                    nav.night_granules_async(bbox=bbox, nominal_date=nominal_date, cmask=cmask,
                                             progress=progress)
                ))

        [found_granules.update(t.result()) for t in tasks]

    except ExceptionGroup as eg:
        for e in eg.exceptions:
            logger.error(f"❌ Sub-task failed: {e}")

    finally:
        if progress and progress_task is not None:
            progress.remove_task(progress_task)




    if cmask:
        cloud_coverage_results = cloud_coverage_batch(urls=list(found_granules.keys()), bbox=bbox, progress=progress)
        for cm_url, g in found_granules.items():
            cloud_cover = cloud_coverage_results[cm_url]
            if cloud_cover is None:cloud_cover = 'Not Available'
            if isinstance(cloud_cover, Exception):
                continue
            g.cloud_cover = cloud_cover
            g.url = cm_url
            selected_granules.append(g)
        selected_granules.sort(key=lambda g: g.rank, reverse=True)

    else:
        selected_granules = list(found_granules.values())
        selected_granules.sort(key=lambda g: g.rank, reverse=True)


    return selected_granules

if __name__ == '__main__':
    import asyncio
    from datetime import datetime
    # import os
    # os.environ['UV_ENV_FILE'] = '/home/work/py/geo-cb-surge/.env'
    # from ntl.io.operational import download
    # from ntl.search.cmr import fetch
    # from ntl.utils import vis
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.DEBUG,  # Or whatever level you use
        format="%(message)s",  # RichHandler handles the timestamps and formatting natively
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)]
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logger.name = 'ntloper'
    events = {
        'Tehran': ('28-02-2026', (50.8, 35.3, 51.9, 36)),  # bombing
        'Abuja': ('23-01-2026', (7, 8.5, 7.8, 9.4)),  # grid failure
        'Dominican Rep': ('23-02-2026', (-72.00, 17.50, -68.30, 20.00)),  # national grid failure
        'Kharkiv/Dnipro': ('26-01-2026', (34.4, 48.2, 38.3, 50.6)),  # Kharkiv/Dnipro grid attacks
        'Porto Rico': ('26-09-2017', (-67.8, 17.6, -65.2, 18.6)),
        'Bahia Blanca': ('18-12-2023', (-62.4, -38.8, -62.2, -38.6))
    }

    site = 'Abuja'
    datestr, bbox = events[site]
    event_date = datetime.strptime(datestr, '%d-%m-%Y')
    target_date = event_date + timedelta(days=1)

    # resolution = 750
    #
    # dst_dir = '/tmp'
    # files = [e for e in os.scandir(dst_dir) if e.name.endswith('.h5') or e.name.endswith('.nc')]
    # print(files)
    # cmask_file_path = None
    # for e in files:
    #     if e.name.startswith('SVDNB'): sdr_file_path = e.path
    #     if e.name.startswith('GDNBO'): geolocation_file_path = e.path
    #     if e.name.startswith('JRR-Cloud'): cmask_file_path = e.path
    #     if '46A3' in e.name: baseline_file_path = e.path
    # if not cmask_file_path:
    with Progress(disable=False, transient=False) as progress:

        for s in VIIRSNavigator.SATELLITES:
            n = VIIRSNavigator(satellite=s)
            print(n)
            phase = n.get_phase_for_date(target_date)
            new_phase = n.get_phase_for_date(target_date=target_date)
            print(s, phase, new_phase)
            break

        # granules = asyncio.run(async_search_granules(  satellites=['SNPP'],
        #     nominal_date=target_date, bbox=bbox, cmask=True, progress=progress
        # ))

        # for g in granules:
        #     print(g)






