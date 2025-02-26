import logging
import os.path
from abc import abstractmethod
from typing import List
from typing import Optional, Union

from osgeo import gdal
from pydantic import BaseModel, FilePath
from sympy.parsing.sympy_parser import parse_expr

from cbsurge.project import Project

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

    @abstractmethod
    def download(self, **kwargs):
        pass

    @abstractmethod
    def resolve(self, **kwargs):
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


