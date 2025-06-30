
import logging
import os.path
import re
from abc import abstractmethod
from typing import List
from typing import Optional, Union

from osgeo import gdal
from pydantic import BaseModel, FilePath
from sympy.parsing.sympy_parser import parse_expr
from rapida.constants import ARROWTYPE2OGRTYPE

from rapida.project.project import Project

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
    source_column: str = None
    source_column_value: str = None
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
    @abstractmethod
    def __call__(self,  **kwargs):
        pass

    def interpolate_template(self, template=None, **kwargs):
        """
        Interpolate values from kwargs into template
        """
        kwargs['country_lower'] = kwargs['country'].lower()
        template_vars = set(re.findall(self._extractor_, template))

        if template_vars:
            for template_var in template_vars:
                if not template_var in kwargs:
                    assert hasattr(self,template_var), f'"{template_var}"  is required to generate source files'

            return template.format(**kwargs)
        else:
            return template



