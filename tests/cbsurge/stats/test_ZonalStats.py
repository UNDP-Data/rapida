import os
import pytest
from cbsurge.stats.ZonalStats import ZonalStats

TEST_DIR = "./tests/cbsurge/stats"

def get_test_path(relative_path):
    """Helper function to generate absolute paths for test assets."""
    return os.path.abspath(f"{TEST_DIR}/{relative_path}")

@pytest.mark.parametrize(
    "input_file, input_raster, operations, operation_cols, expected_columns, output_file, num_columns",
    [
        # Test case 1: Single operation
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif"],
            ["sum"],
            None,
            ["rwa_m_5_2020_constrained_UNadj_sum"],
            "assets/admin2_stats.fgb",
            10,
        ),
        # Test case 2: Multiple operations
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif"],
            ["sum", "count"],
            None,
            ["rwa_m_5_2020_constrained_UNadj_sum", "rwa_m_5_2020_constrained_UNadj_count"],
            None,
            11,
        ),
        # Test case 3: Custom column names
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif"],
            ["sum", "median"],
            ["male_5_age_sum", "memale_5_age_median"],
            ["male_5_age_sum", "memale_5_age_median"],
            None,
            11,
        ),
        # Test case 4: Output GPKG
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif"],
            ["sum", "median"],
            None,
            ["rwa_m_5_2020_constrained_UNadj_sum", "rwa_m_5_2020_constrained_UNadj_median"],
            "assets/admin2_stats.gpkg",
            11,
        ),
        # Test case 5: multi rasters with single operation
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif", "assets/rwa_f_5_2020_constrained_UNadj.tif"],
            ["sum"],
            None,
            ["rwa_m_5_2020_constrained_UNadj_sum", "rwa_f_5_2020_constrained_UNadj_sum"],
            "assets/admin2_stats_multi.fgb",
            11,
        ),
        # Test case 6: multi rasters with multi operations
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif", "assets/rwa_f_5_2020_constrained_UNadj.tif"],
            ["sum", "median"],
            None,
            ["rwa_m_5_2020_constrained_UNadj_sum", "rwa_m_5_2020_constrained_UNadj_median", "rwa_f_5_2020_constrained_UNadj_sum", "rwa_f_5_2020_constrained_UNadj_median"],
            "assets/admin2_stats_multi.fgb",
            13,
        ),
        # Test case 7: multi rasters with single operation and custom column names
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif", "assets/rwa_f_5_2020_constrained_UNadj.tif"],
            ["sum"],
            ["male 5 sum", "female 5 sum"],
            ["male 5 sum", "female 5 sum"],
            "assets/admin2_stats_multi.fgb",
            11,
        ),
        # Test case 8: multi rasters with multi operations
        (
            "assets/admin2.geojson",
            ["assets/rwa_m_5_2020_constrained_UNadj.tif", "assets/rwa_f_5_2020_constrained_UNadj.tif"],
            ["sum", "median"],
            ["male 5 sum", "male 5 median", "female 5 sum", "female 5 median"],
            ["male 5 sum", "male 5 median", "female 5 sum", "female 5 median"],
            "assets/admin2_stats_multi.fgb",
            13,
        ),
    ]
)
def test_zonal_stats(input_file, input_raster, operations, operation_cols, expected_columns, output_file, num_columns):
    """Test raster files with various operations and configurations."""
    input_file = get_test_path(input_file)
    for index, raster in enumerate(input_raster):
        input_raster[index] = get_test_path(input_raster[index])
    output_file = get_test_path(output_file) if output_file else None

    with ZonalStats(input_file, target_crs="ESRI:54009") as st:
        gdf = st.compute(
            input_raster,
            operations=operations,
            operation_cols=operation_cols)

        for col in expected_columns:
            assert col in gdf.columns, f"Column '{col}' not found in result."

        assert len(gdf.columns) == num_columns, f"The number of columns {len(gdf.columns)} does not match the expected number of columns."

        if output_file:
            st.write(output_file, target_crs="EPSG:3857")
            assert os.path.exists(output_file), f"Output file {output_file} not found."
            if os.path.exists(output_file):
                os.remove(output_file)
