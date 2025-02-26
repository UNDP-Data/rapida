import asyncio
import concurrent
import logging
import os.path
import random
import threading
import time
from abc import abstractmethod
from collections import deque
from typing import List
from typing import Optional, Union
import shapely
from osgeo import gdal, ogr, osr
from pyarrow import compute as pc
from pydantic import BaseModel, FilePath
from pyogrio import read_info
from rich.progress import Progress
from sympy.parsing.sympy_parser import parse_expr

from cbsurge.constants import ARROWTYPE2OGRTYPE
from cbsurge.project import Project
from cbsurge.util.downloader import downloader

logger = logging.getLogger(__name__)
gdal.UseExceptions()

class Variable(BaseModel):
    name: str
    title: str
    source: Optional[str] = None
    sources: Optional[Union[List[str], str]] = None
    dep_vars: Optional[List[str]] = None
    local_path: FilePath = None
    component: str = None
    operator: str = None
    _extractor_: str = r"\{([^}]+)\}"
    _default_operators_ = '+-/*%'
    _source_folder_: str = None

    def __init__(self, **kwargs):
        """
        Initialize the object with the provided arguments.
        """

        super().__init__(**kwargs)
        self.__dict__.update(kwargs)

        try:
            parsed_expr = parse_expr(self.sources)

            self.dep_vars = [s.name for s in parsed_expr.free_symbols]

        except (SyntaxError, AttributeError):
            pass
        project = Project(path=os.getcwd())
        self._source_folder_ = os.path.join(project.data_folder, self.component, self.name)


    def __str__(self):
        """
        String representation of the class.
        """
        return f'{self.__class__.__name__} {self.model_dump_json(indent=2)}'

    @abstractmethod
    def compute(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    def __call__(self,  **kwargs):
        """
        Assess a variable. Essentially this means a series of steps in a specific order:
            - download
            - preprocess
            - analysis/zonal stats

        :param kwargs:
        :return:
        """

        force_compute = kwargs.get('force_compute', False)
        if not self.dep_vars: #simple variable,
            if not force_compute:
                # logger.debug(f'Downloading {self.name} source')
                self.download(**kwargs)
            else:
                # logger.debug(f'Computing {self.name} using gdal_calc from sources')
                self.compute(**kwargs)
        else:
            if self.operator:
                if not force_compute:
                    # logger.debug(f'Downloading {self.name} from  source')
                    self.download(**kwargs)
                else:
                    # logger.debug(f'Computing {self.name}={self.sources} using GDAL')
                    self.compute(**kwargs)
            else:
                #logger.debug(f'Computing {self.name}={self.sources} using GeoPandas')
                sources = self.resolve(**kwargs)

        return self.evaluate(**kwargs)


