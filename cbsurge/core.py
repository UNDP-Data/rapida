import json
import os.path
import sys
import asyncio
from sympy.parsing.sympy_parser import parse_expr
from osgeo_utils import gdal_calc
from pydantic import BaseModel, FilePath
from typing import Optional, List, Union
import re

from cbsurge.az.blobstorage import download
from cbsurge.session import Session
from cbsurge import util
import logging
from cbsurge.az import blobstorage



logger = logging.getLogger(__name__)



# class Variable(BaseModel):
#     name: str
#     title: str
#     source: Optional[str] = None
#     files: Optional[List[str]] = None
#     #is_computable: Optional[bool] = None  # Optional field, default True
#     is_raster: Optional[bool] = None  # Optional field, default False
#
#     _extractor_: str = r"\{([^}]+)\}"
#     _default_operators_ = '+-/*%'
#
#     def __new__(cls, *args, **kwargs):
#         """
#         A __new__ method that uses 'type()' with 3 arguments to dynamically create
#         a class with the given attributes.
#         """
#         ann = cls.__annotations__.copy()
#         files = kwargs.get('files', None)
#         module_name = sys.modules[__name__].__name__
#         #source = kwargs.get('source', None)
#         if files is not None:
#             kwargs['is_raster'] = True
#             class_name = f'{module_name}.RasterVariable'
#         else:
#             kwargs['is_raster'] = False
#             class_name = f'{module_name}.VectorVariable'
#             kwargs['files'] = None
#
#         # Creating a new class using 'type()' with 3 arguments
#         new_class = type(
#             class_name,               # Class name (using the current class name)
#             (cls,),                      # Inherit from the current class
#             {**kwargs, '__annotations__': ann}                  # Add attributes from kwargs to the class
#         )
#         return super().__new__(new_class)
#
#     def __init__(self, **kwargs):
#         """
#         Initialize the object with the provided arguments.
#         """
#         super().__init__(**kwargs)
#         self.resolve(**kwargs)
#
#     def resolve_file(self, **kwargs):
#         """
#         Resolve file paths with the provided kwargs, ensuring that the necessary variables are included.
#         """
#         if 'iso3_country_code' in kwargs:
#             kwargs['iso3_country_code_lower'] = kwargs['iso3_country_code_lower'].lower()
#         variables = set(re.findall(self._extractor_, self.source))
#         for varname in variables:
#             assert varname in kwargs, f'"{varname}" kwarg is required to generate source files'
#         return self.source.format(**kwargs)
#
#     def download(self, **kwargs):
#         """
#         Placeholder method for downloading logic.
#         """
#         pass
#
#     def compute(self):
#         """
#         Placeholder method for parsing and calculating expressions based on 'source'.
#         """
#         if self.source:
#             parsed_expr = parse_expr(self.source)
#             variables = parsed_expr.free_symbols
#
#     def __call__(self, *args, **kwargs):
#         """
#         NotImplemented error for call functionality, subclasses should implement this.
#         """
#         raise NotImplementedError(f'Only subclasses of {self.__class__.__name__} implement this method')
#
#     def resolve(self, **kwargs):
#         """
#         Resolving logic for optional fields like 'is_computable' based on the presence of operators.
#         """
#         if self.source is not None:
#             operators = set(self._default_operators_).intersection(self.source)
#             if operators:
#                 pass
#         else:
#             self.is_computable = False
#
#     def __str__(self):
#         """
#         String representation of the class.
#         """
#         return f'{self.__class__.__name__} {self.model_dump_json(indent=2)}'
#

class SurgeVariable(BaseModel):
    name: str
    title: str
    source: Optional[str] = None
    sources: Optional[Union[List[str], str]] = None
    variables: Optional[List[str]] = None
    local_path: FilePath = None
    component: str = None
    _extractor_: str = r"\{([^}]+)\}"
    _default_operators_ = '+-/*%'
    _source_folder_: str = None

    def __init__(self, **kwargs):
        """
        Initialize the object with the provided arguments.
        """
        super().__init__(**kwargs)

        try:
            parsed_expr = parse_expr(self.sources)
            self.variables = [s.name for s in parsed_expr.free_symbols]

        except (SyntaxError, AttributeError):
            pass
        with Session() as s:
            root_folder = s.get_root_data_folder()
            self._source_folder_ = os.path.join(root_folder, self.component, self.name)


    def __str__(self):
        """
        String representation of the class.
        """
        return f'{self.__class__.__name__} {self.model_dump_json(indent=2)}'

    def download(self, **kwargs):

            if not os.path.exists(self._source_folder_):
                os.makedirs(self._source_folder_)
            src_path = self.interpolate(template=self.source, **kwargs)
            _, file_name = os.path.split(src_path)
            self.local_path = os.path.join(self._source_folder_, file_name)
            logger.info(f'Going to download {src_path} to {self.local_path} ')
            downloaded_file = asyncio.run(blobstorage.download_blob(src_path=src_path,dst_path=self.local_path)                             )
            assert downloaded_file == self.local_path, f'The local_path differs from {downloaded_file}'
            return downloaded_file

    def compute(self, **kwargs):
        pass

    def resolve(self, **kwargs):
        for var in self.variables:
            print(var)

    def __call__(self,  **kwargs):

            # # first try to download from source
            # if self.source:
            #     self.download(**kwargs)
            # use sources to compute
            if self.sources:
                if self.variables:
                    logger.info(f'Going to compute {self.name} from {self.sources}')
                    self.resolve()

                else:
                    logger.info(f'Going to sum up {self.name} from source files')
                    # interpolate templates
                    source_blobs = list()
                    for source_template in self.sources:
                        source_blobs.append(self.interpolate(template=source_template, **kwargs))
                    downloaded_files = asyncio.run(
                        blobstorage.download_blobs(src_blobs=source_blobs, dst_folder=self._source_folder_)
                    )




    def interpolate(self, template=None, **kwargs):
        """
        Resolve file paths with the provided kwargs, ensuring that the necessary variables are included.
        """
        if 'country' in kwargs:
            kwargs['country_lower'] = kwargs['country'].lower()
        template_vars = set(re.findall(self._extractor_, template))
        for template_var in template_vars:
            assert template_var in kwargs, f'"{template_var}"  is required to generate source files'
        return template.format(**kwargs)




if __name__ == '__main__':
    logger = util.setup_logger(name='rapida', level=logging.INFO)
    with Session() as ses:
        for var_name, var_data in ses.config['variables']['population'].items():

            v = SurgeVariable(name=var_name, component='population', **var_data)
            if not v.variables:
                continue
            print(var_name)
            r = v(year=2020, country='MDA')
            #print(v)
            break

