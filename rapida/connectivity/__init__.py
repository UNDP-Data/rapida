import datetime

import os.path
from unittest.mock import inplace

from rapida.util.bbox_param_type import get_best_semantic_label
import geopandas as gpd
import pandas as pd
import logging
from rich.progress import Progress
from rapida.connectivity.io import (prepare_osm_pbf,extract_health_sites,
                                    extract_origins_from_geojson, extract_origins,
                                    extract_water_bodies, extract_roads, filter_polygons)
from rapida.connectivity.graph import compile_valhalla_graph
from rapida.connectivity.isochrone import connectivity_areas
from rapida.cli.assess import assess
import click
from rapida.project.project import Project
from tempfile import TemporaryDirectory
import gc
from math import nan
import pyogrio

logger = logging.getLogger(__name__)






async def run_connectivity_analysis(
        bbox:tuple[float, float, float, float]=None, travel_mode:str=None, time_intervals:list[int] =None,
        dst_dir:str=None, barriers_dataset:str=None, barriers_layer:str=None, barriers_buffer:int=None,
        sites_dataset:str=None, sites_layer:str=None,pop_vars:str|tuple[str]=None, stats_admin_level:int=None,
        progress:Progress=None, year=datetime.datetime.now().year, disjoint:bool=False, radius:float=None,
        clip_country:str=None, smooth:bool=False
    ):
    if bbox is None:
        assert sites_dataset is not None, f'site_dataset has to be provided when bbox is not'
        try:
            slayer = int(sites_layer)
        except ValueError:
            slayer = sites_layer
        gdf = gpd.read_file(sites_dataset, layers=slayer)
        if not gdf.crs.is_geographic:
            gdf.to_crs('EPSG:4326', inplace=True)
        bbox = gdf.total_bounds
    bbox_label = get_best_semantic_label(bbox=bbox)
    dest_dir = os.path.join(dst_dir, bbox_label)
    bbox_pbf = await prepare_osm_pbf(bbox=bbox, dst_dir=dest_dir, progress=progress, clip_country=clip_country)


    if sites_dataset is None:
        sites = await extract_health_sites(pbf_path=bbox_pbf, dst_dir=dest_dir, progress=progress)
    else:
        sites = sites_dataset

    dag_tar_path = await compile_valhalla_graph(pbf_path=bbox_pbf,dst_dir=dest_dir, progress=progress)
    origins = extract_origins(sites_dataset=sites, src_layer=sites_layer)

    logger.info(f'Computing isochrones for {len(origins)} sites')
    isochrones_gdf = await connectivity_areas(
        tar_path=dag_tar_path, origins=origins, travel_mode=travel_mode, intervals_minutes=time_intervals,
        radius=radius, disjoint=disjoint, smooth=smooth)

    if clip_country:
        logger.info('Clipping isochrones with ADM0')
        url = f"/vsicurl/https://undpngddlsgeohubdev01.blob.core.windows.net/admin/cgaz/geoBoundariesCGAZ_ADM0.fgb"
        a0_gdf = gpd.read_file(url, bbox=bbox, engine="pyogrio")
        if not clip_country in a0_gdf['iso3'].tolist():
            raise Exception(f'--clip-country {clip_country} does not intersect bbox {bbox}')
        a0_gdf = a0_gdf[a0_gdf['iso3'] == clip_country]
        if isochrones_gdf.crs != a0_gdf.crs:
            a0_gdf = a0_gdf.to_crs(isochrones_gdf.crs)
        isochrones_gdf = isochrones_gdf.clip(a0_gdf)
        isochrones_gdf['iso3'] = clip_country

    water_bodies_path = await extract_water_bodies(pbf_path=bbox_pbf,dst_dir=dest_dir, progress=progress)
    water_gdf = gpd.read_file(water_bodies_path)
    if not water_gdf.empty:
        logger.info('Removing water bodies from isochrones')
        if isochrones_gdf.crs != water_gdf.crs:
            water_gdf = water_gdf.to_crs(isochrones_gdf.crs)

        water_poly = water_gdf.geometry.union_all()
        isochrones_gdf["geometry"] = isochrones_gdf.geometry.difference(water_poly)

        # 3. Clean up empty/exploded geometries if any were cut into pieces
        isochrones_gdf = isochrones_gdf[~isochrones_gdf.is_empty].explode(index_parts=False)
        #TD
        # ADD THIS: Dissolve by contour to merge overlaps from different sites
        isochrones_gdf = isochrones_gdf.dissolve(by='contour', as_index=False)


    # 5. Save the final processed isochrones to GeoJSON
    isochrones_path = os.path.join(dest_dir, "isochrones.geojson")

    isochrones_gdf.to_file(isochrones_path, driver="GeoJSON", engine='pyogrio')


    if pop_vars:
        logger.info(f'Computing zonal stats for regular isochrones')
        if stats_admin_level:
            url = f"/vsicurl/https://undpngddlsgeohubdev01.blob.core.windows.net/admin/cgaz/geoBoundariesCGAZ_ADM{stats_admin_level}.fgb"
            adm_gdf = gpd.read_file(url, bbox=bbox, engine="pyogrio")
            if clip_country:
                adm_gdf = adm_gdf[adm_gdf['iso3'] == clip_country]
            if 'iso3' in isochrones_gdf.columns.tolist():
                adm_gdf.drop(columns=['iso3'], inplace=True)
            if isochrones_gdf.crs != adm_gdf.crs:
                adm_gdf.to_crs(isochrones_gdf.crs, inplace=True)
            results = []
            for i, unit in adm_gdf.iterrows():
                unit_gdf = gpd.GeoDataFrame([unit], crs=adm_gdf.crs, geometry=adm_gdf.geometry.name)
                # 2. Fast pre-filter: Skip admin units that don't even touch the isochrones' bounding boxes
                if isochrones_gdf.sindex.intersection(unit_gdf.total_bounds).size == 0:
                    continue
                # 3. Run the overlay chunk
                chunk_result = gpd.overlay(isochrones_gdf, unit_gdf, how="intersection", keep_geom_type=True)

                if not chunk_result.empty:
                    results.append(chunk_result)

            # 3. Recombine into the final Spatially-enabled DataFrame
            split_isochrones = gpd.GeoDataFrame(
                pd.concat(results, ignore_index=True),
                crs=isochrones_gdf.crs
            )
            split_isochrones.to_file(isochrones_path, driver="GeoJSON")
            del split_isochrones
            del isochrones_gdf
            del results

        with TemporaryDirectory(dir=dest_dir, delete=True) as project_folder:
            project = Project(path=project_folder, polygons=isochrones_path, comment='temp project for conn isochrones')
            with click.Context(assess) as ctx:
                ctx.ensure_object(dict)
                ctx.obj['progress'] = progress
                # 2. Use invoke. Do NOT pass 'ctx' manually here.
                # Click intercepts this and injects it as the first argument automatically.
                ctx.invoke(
                    assess,
                    components=('population',),
                    variables=pop_vars,
                    year=year,
                    project=project.path,
                    force=False
                )
                stat_gpkg_path = os.path.join(project_folder,'data', f'{project.name}.gpkg')
                pop_stat_gdf = gpd.read_file(stat_gpkg_path, layer='stats.population')
                if not disjoint:
                    pop_stat_gdf = pop_stat_gdf.iloc[pop_stat_gdf.geometry.area.sort_values(ascending=False).index]
                pop_stat_gdf = pop_stat_gdf.to_crs('EPSG:4326')

                pop_stat_gdf.to_file(
                    filename=isochrones_path,
                    driver="GeoJSON",
                    engine="pyogrio",
                    mode="w",
                    layer='isochrones',
                    promote_to_multi=True,
                    index=False
                )


    if barriers_dataset is not None and pop_vars:
        info = pyogrio.read_info(barriers_dataset)
        if 'polygon' in info['geometry_type'].lower(): # keep only polys that actually intersect the roads
            logger.info(f'Removing barrier polygons that do not intersect roads...')
            roads_dataset = await extract_roads(pbf_path=bbox_pbf, dst_dir=dest_dir, progress=progress)
            barriers_dataset = filter_polygons(poly_ds_path=barriers_dataset, lines_ds_path=roads_dataset, dst_dir=dest_dir)

        logger.info(f'Computing isochrones with barriers')
        barrier_isochrones_gdf = await connectivity_areas(
        tar_path=dag_tar_path, origins=origins, travel_mode=travel_mode, intervals_minutes=time_intervals,
        barriers_dataset=barriers_dataset, barriers_layer=barriers_layer, barriers_buffer=barriers_buffer, disjoint=disjoint, radius=radius,
            smooth=smooth, progress=progress
                             )

        if clip_country:
            logger.info('Clipping barriers isochrones with ADM0')
            url = f"/vsicurl/https://undpngddlsgeohubdev01.blob.core.windows.net/admin/cgaz/geoBoundariesCGAZ_ADM0.fgb"
            a0_gdf = gpd.read_file(url, bbox=bbox, engine="pyogrio")
            if not clip_country in a0_gdf['iso3'].tolist():
                raise Exception(f'--clip-country {clip_country} does not intersect bbox {bbox}')
            a0_gdf = a0_gdf[a0_gdf['iso3'] == clip_country]
            if barrier_isochrones_gdf.crs != a0_gdf.crs:
                a0_gdf = a0_gdf.to_crs(barrier_isochrones_gdf.crs)
            barrier_isochrones_gdf = barrier_isochrones_gdf.clip(a0_gdf)
            barrier_isochrones_gdf['iso3'] = clip_country
        if not water_gdf.empty:
            logger.info('Removing water bodies from barrier isochrones')
            barrier_isochrones_gdf["geometry"] = barrier_isochrones_gdf.geometry.difference(water_poly)
            #barrier_isochrones_gdf = barrier_isochrones_gdf.overlay(water_gdf, how='difference')
            # 3. Clean up empty/exploded geometries if any were cut into pieces
            barrier_isochrones_gdf = barrier_isochrones_gdf[~barrier_isochrones_gdf.is_empty].explode(index_parts=False)
            # ADD THIS: Dissolve by contour to merge overlaps from different sites
            barrier_isochrones_gdf = barrier_isochrones_gdf.dissolve(by='contour', as_index=False)


        barrier_isochrones_path = os.path.join(dest_dir, 'isochrones_with_barriers.geojson')
        barrier_isochrones_gdf.to_file(barrier_isochrones_path, driver="GeoJSON", engine="pyogrio")
        if pop_vars:
            logger.info(f'Computing zonal stats for barrier isochrones')

            if stats_admin_level:
                url = f"/vsicurl/https://undpngddlsgeohubdev01.blob.core.windows.net/admin/cgaz/geoBoundariesCGAZ_ADM{stats_admin_level}.fgb"
                adm_gdf = gpd.read_file(url, bbox=bbox, engine="pyogrio")
                if clip_country:
                    adm_gdf = adm_gdf[adm_gdf['iso3'] == clip_country]
                if 'iso3' in barrier_isochrones_gdf.columns.tolist():
                    adm_gdf.drop(columns=['iso3'], inplace=True)
                if barrier_isochrones_gdf.crs != adm_gdf.crs:
                    adm_gdf.to_crs(barrier_isochrones_gdf.crs, inplace=True)
                results = []
                for i, unit in adm_gdf.iterrows():
                    unit_gdf = gpd.GeoDataFrame([unit], crs=adm_gdf.crs, geometry=adm_gdf.geometry.name)
                    # 2. Fast pre-filter: Skip admin units that don't even touch the isochrones' bounding boxes
                    if barrier_isochrones_gdf.sindex.intersection(unit_gdf.total_bounds).size == 0:
                        continue
                    # 3. Run the overlay chunk
                    chunk_result = gpd.overlay(barrier_isochrones_gdf, unit_gdf, how="intersection", keep_geom_type=True)

                    if not chunk_result.empty:
                        results.append(chunk_result)

                # 3. Recombine into the final Spatially-enabled DataFrame
                split_barrier_isochrones = gpd.GeoDataFrame(
                    pd.concat(results, ignore_index=True),
                    crs=barrier_isochrones_gdf.crs
                )


                split_barrier_isochrones.to_file(barrier_isochrones_path, driver="GeoJSON", engine='pyogrio', promote_to_multi=True,
                                         index=False)

                del barrier_isochrones_gdf
                del split_barrier_isochrones
                del results


            with TemporaryDirectory(dir=dest_dir, delete=True) as project_folder:
                project = Project(path=project_folder, polygons=barrier_isochrones_path, comment='temp project for conn isochrones')
                with click.Context(assess) as ctx:
                    ctx.ensure_object(dict)
                    ctx.obj['progress'] = progress
                    # 2. Use invoke. Do NOT pass 'ctx' manually here.
                    # Click intercepts this and injects it as the first argument automatically.
                    ctx.invoke(
                        assess,
                        components=('population',),
                        variables=pop_vars,
                        year=year,
                        project=project.path,
                        force=False
                    )

                    stat_gpkg_path = os.path.join(project_folder, 'data', f'{project.name}.gpkg')
                    barrier_pop_stat_gdf = gpd.read_file(stat_gpkg_path, layer='stats.population')
                    if not disjoint:
                        barrier_pop_stat_gdf = barrier_pop_stat_gdf.iloc[barrier_pop_stat_gdf.geometry.area.sort_values(ascending=False).index]
                    barrier_pop_stat_gdf = barrier_pop_stat_gdf.to_crs('EPSG:4326')

                    if not stats_admin_level:
                        barrier_pop_stat_gdf.to_file(
                            filename=barrier_isochrones_path,
                            driver="GeoJSON",
                            engine="pyogrio",
                            mode="w",
                            layer='barrier_isochrones',
                            promote_to_multi=True,
                            index=False
                        )
                    else:

                        logger.info('Aggregating zonal stats independently and pivoting to wide format...')

                        pop_col_names = [f'{popv}_{year}' for popv in pop_vars]
                        new_pop_col_names = [f'{popv}_{year}_barrier' for popv in pop_vars]
                        col_name_dict = dict(zip(pop_col_names, new_pop_col_names))
                        admin_col_name = f'admin{stats_admin_level}_name'
                        assert admin_col_name in pop_stat_gdf.columns.tolist()

                        pop_stat_gdf['contour'] = pop_stat_gdf['contour'].astype(float).astype(int)
                        barrier_pop_stat_gdf['contour'] = barrier_pop_stat_gdf['contour'].astype(float).astype(int)

                        pop_stat_gdf[admin_col_name] = pop_stat_gdf[admin_col_name].astype(str).str.strip()
                        barrier_pop_stat_gdf[admin_col_name] = barrier_pop_stat_gdf[admin_col_name].astype(
                            str).str.strip()

                        # 1. Group and sum the base stats independently (vectorized aggregation)
                        base_agg = pop_stat_gdf.groupby([admin_col_name, 'contour'])[pop_col_names].sum().reset_index()

                        # 2. Group and sum the barrier stats independently (vectorized aggregation)
                        barrier_agg = barrier_pop_stat_gdf.groupby([admin_col_name, 'contour'])[
                            pop_col_names].sum().reset_index()
                        barrier_agg.rename(columns=col_name_dict, inplace=True)

                        # 3. Merge the tiny aggregated tables
                        # outer join ensures we don't lose contours if one dataset has contours the other doesn't
                        grouped_df = base_agg.merge(barrier_agg, on=[admin_col_name, 'contour'], how='outer')

                        # 4. Handle NAs and compute differences vector-wise
                        val_columns = []
                        for pvar, bar_pvar in col_name_dict.items():
                            grouped_df[pvar] = grouped_df[pvar].fillna(0)
                            grouped_df[bar_pvar] = grouped_df[bar_pvar].fillna(0)

                            # Clean column names for the differences
                            diff_col = f'{pvar}_diff'
                            perc_col = f'{pvar}_perc_diff'

                            # Calculate differences
                            grouped_df[diff_col] = (grouped_df[pvar] - grouped_df[bar_pvar]).clip(lower=0)

                            # Calculate percentage (safeguarding against division by zero)
                            safe_div = grouped_df[pvar].replace(0, nan)
                            grouped_df[perc_col] = (grouped_df[diff_col] / safe_div).fillna(0) * 100

                            val_columns.extend([pvar, bar_pvar, diff_col, perc_col])

                        # 5. Pivot to Wide Format (Option 2: contours become columns)
                        pivot_df = grouped_df.pivot(index=admin_col_name, columns='contour', values=val_columns)

                        # Flatten the MultiIndex columns (e.g., ('male_total_2026', 15.0) -> 'male_total_2026_15min')
                        pivot_df.columns = [f"{col[0]}_{int(col[1])}min" for col in pivot_df.columns]
                        wide_df = pivot_df.reset_index()

                        # 6. Attach geometries from the original adm_gdf
                        logger.info('Merging aggregated stats back onto original admin boundaries...')

                        # Handle case where CGAZ admin column might natively be 'shapeName'
                        adm_join_col = admin_col_name if admin_col_name in adm_gdf.columns else 'shapeName'

                        final_gdf = adm_gdf.merge(
                            wide_df,
                            left_on=adm_join_col,
                            right_on=admin_col_name,
                            how='inner'  # Use 'inner' to only keep admin units that actually had isochrones
                        )

                        gc.collect()

                        with TemporaryDirectory(dir=dest_dir, delete=True) as admin_project_folder:
                            logger.info(f'Computing zonal stats for total population ')
                            adm_ds_path = os.path.join(dest_dir, f'admin_{stats_admin_level}.fgb')
                            adm_gdf.to_file(adm_ds_path, driver="FlatGeobuf", engine="pyogrio")
                            admin_project = Project(path=admin_project_folder, polygons=adm_ds_path,
                                              comment='temp project for admin stats')
                            with click.Context(assess) as ctx:
                                ctx.ensure_object(dict)
                                ctx.obj['progress'] = progress
                                # 2. Use invoke. Do NOT pass 'ctx' manually here.
                                # Click intercepts this and injects it as the first argument automatically.
                                ctx.invoke(
                                    assess,
                                    components=('population',),
                                    variables=['total'],
                                    year=year,
                                    project=admin_project.path,
                                    force=False
                                )

                                admin_stat_gpkg_path = os.path.join(admin_project_folder, 'data', f'{admin_project.name}.gpkg')
                                admin_pop_stat_gdf = gpd.read_file(admin_stat_gpkg_path, layer='stats.population')
                                final_gdf = final_gdf.merge(admin_pop_stat_gdf[[admin_col_name, f'total_{year}']],on=admin_col_name)

                            if os.path.exists(adm_ds_path):os.remove(adm_ds_path)

                        admin_iso_stats = os.path.join(dest_dir, f"admin{stats_admin_level}_iso_stats.geojson")
                        logger.info(f"Writing final aggregated admin boundaries to {admin_iso_stats}")

                        final_gdf.to_file(
                            filename=admin_iso_stats,  # Fixed: this was pointing to barrier_isochrones_path
                            driver="GeoJSON",
                            engine="pyogrio",
                            mode="w",
                            layer=f"admin{stats_admin_level}_iso_stats",
                            promote_to_multi=True,
                            index=False
                        )

    return