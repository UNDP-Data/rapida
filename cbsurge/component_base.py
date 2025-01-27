from abc import ABCMeta, abstractmethod
from typing import List

class ComponentBase(metaclass=ABCMeta):
    """
    A base class for a component.
    Each component class should have the following methods to be implemented:

    - download: Download files for a component
    - assess: A command to do all processing for a component including downloading component data, merging and making stats for a given admin data.
    """

    @abstractmethod
    def get_available_variables(self) -> List[str]:
        """
        Returns a list of available variables for this component.
        """
        pass

    @abstractmethod
    def download(self,
                 output_folder: str,
                 bbox: List[float] = None,
                 countries: List[str] = None,
                 variables: List[str] = None
                 )->List[str]:
        """
        Download files for a component. Either `bbox` or `countries` must be provided.

        Args:
            output_folder: output folder to store downloaded files
            bbox: The bounding box for an interested area (the list of float values: minx, miny, maxx, maxy)
            countries: List of country ISO 3 codes.
            variables: Optional. The list of names of a variable to download. If skipped, all variables are downloaded.
        Returns:
            the list of downloaded file paths
        """
        pass

    @abstractmethod
    def assess(self,
               admin_file: str,
               output_folder: str,
               mask_file: str = None,
               masked_value: float = None,
               variables: List[str] = None
               )->str:
        """
        Do all processing for a component including downloading component data, merging and making stats for a given admin data.

        Args:
            admin_file: admin vector file path. Interested area either BBOX or country ISO3 codes are computed from a given admin file
            output_folder: output folder to store all intermediary files and output files
            mask_file: optional. raster mask file path. If provided, mask all component variables to compute affected variables. Affected variables will be named like `affected_XXXX`.
            masked_value: optional. As default, 1 is used to mask unless masked value is provided through the argument.
            variables: optional. The list of names of a variable to process. If skipped, all variables are processed.
        Returns:
            A file path of output file
        """
        pass