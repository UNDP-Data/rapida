import os
import pytest
from cbsurge.stats.ZonalStats import ZonalStats

# To test this file, execute the below command
# PYTHONPATH=$(pwd) pytest cbsurge/stats -v

TEST_DIR = "./cbsurge/stats/tests"

def get_test_path(relative_path):
    """Helper function to generate absolute paths for test assets."""
    return os.path.abspath(f"{TEST_DIR}/{relative_path}")

@pytest.mark.parametrize(
    "input_file, input_raster, operations, expected_columns, output_file, num_columns",
    [
        # Test case 1: Single operation
        (
            "assets/admin2.geojson",
            "assets/rwa_m_5_2020_constrained_UNadj.tif",
            ["sum"],
            ["rwa_m_5_2020_constrained_UNadj_sum"],
            "assets/admin2_stats.fgb",
            10,
        ),
        # Test case 2: Multiple operations
        (
            "assets/admin2.geojson",
            "assets/rwa_m_5_2020_constrained_UNadj.tif",
            ["sum", "count"],
            ["rwa_m_5_2020_constrained_UNadj_sum", "rwa_m_5_2020_constrained_UNadj_count"],
            None,
            11,
        ),
        # Test case 3: Custom column names
        (
            "assets/admin2.geojson",
            "assets/rwa_m_5_2020_constrained_UNadj.tif",
            ["sum", "median"],
            ["male_5_age_sum", "memale_5_age_median"],
            None,
            11,
        ),
        # Test case 4: Output GPKG
        (
            "assets/admin2.geojson",
            "assets/rwa_m_5_2020_constrained_UNadj.tif",
            ["sum", "median"],
            ["male_5_age_sum", "memale_5_age_median"],
            "assets/admin2_stats.gpkg",
            11,
        ),
    ]
)
def test_zonal_stats(input_file, input_raster, operations, expected_columns, output_file, num_columns):
    """Test raster files with various operations and configurations."""
    input_file = get_test_path(input_file)
    input_raster = get_test_path(input_raster)
    output_file = get_test_path(output_file) if output_file else None

    with ZonalStats(input_file, target_srid=54009) as st:
        gdf = st.compute(input_raster, operations=operations, operation_cols=(expected_columns if len(expected_columns) == len(operations) else None))

        for col in expected_columns:
            assert col in gdf.columns, f"Column '{col}' not found in result."

        assert len(gdf.columns) == num_columns, f"The number of columns {len(gdf.columns)} does not match the expected number of columns."

        if output_file:
            st.write(output_file, target_srid=3857)
            assert os.path.exists(output_file), f"Output file {output_file} not found."
            if os.path.exists(output_file):
                os.remove(output_file)
