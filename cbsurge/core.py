import abc
import os.path
import asyncio
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel, FilePath
from typing import Optional, List, Union
import re
from typing import List, Dict
from abc import abstractmethod
from cbsurge.project import Project
from cbsurge.stats.zst import zonal_stats, sumup
from cbsurge.session import Session
from cbsurge import util
import logging
from cbsurge.az import blobstorage
import importlib
from pkgutil import walk_packages


logger = logging.getLogger(__name__)

def dump_variables(root_package_name = 'cbsurge', variables_module='variables', function_name='generate_variables'):
    """
    Iterate over all modules in path  that start with root_package_name, end with variables_module and are
    not packages. Consequently import the function_name function and run it.

    :param root_package_name: str, cbsurge
    :param variables_module: str, variables
    :param function_name: 'generate_variables'
    :return: dict['variables'] = {'component': component_vars}
    """
    vars_dict = {}
    vars_dict[variables_module] = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for p in walk_packages(path=[dir_path]):
        if p.name.startswith(root_package_name) and p.name.endswith(variables_module) and not p.ispkg:
            m = importlib.import_module(name=p.name)
            if hasattr(m, function_name):
                var_dict = getattr(m, 'generate_variables')()
                component_name = p.name.split('.')[-2]

                vars_dict[variables_module][component_name] = var_dict
    return vars_dict


class Component():
    """
    A base class for a component.
    Each component class should have the following methods to be implemented:

    - download: Download files for a component
    - assess: A command to do all processing for a component including downloading component data, merging and making stats for a given admin data.
    """

    def __init__(self, **kwargs):
        self.component_name = self.__class__.__name__.lower().split('component')[0]
        super().__init__(**kwargs)

    @property
    def variables(self) -> List[str]:
        with Session() as ses:
            return ses.get_variables(component=self.component_name)



    @abstractmethod
    def download(self, variables: List[str] = None, **kwargs) -> List[str]:
        """
        Iterate over variables and download one by one
        :param variables:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def __call__(self, **kwargs):

        pass


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


    def _update_(self, **kwargs):
        args = self.__dict__.copy()
        args.update(kwargs)
        return args

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



    def download(self, **kwargs):

        """Download variable"""
        logger.debug(f'Downloading {self.name}')
        src_path = self.interpolate_template(template=self.source, **kwargs)
        _, file_name = os.path.split(src_path)
        self.local_path = os.path.join(self._source_folder_, file_name)
        if os.path.exists(self.local_path):
            return self.local_path
        if not os.path.exists(self._source_folder_):
            os.makedirs(self._source_folder_)
        logger.info(f'Going to download {self.name} from {src_path}')
        downloaded_file = asyncio.run(blobstorage.download_blob(src_path=src_path,dst_path=self.local_path)                             )
        assert downloaded_file == self.local_path, f'The local_path differs from {downloaded_file}'



    def __call__(self,  **kwargs):
            """
            Assess a variable. Essentially this means a series of steps in a specific order:
                - download
                - preprocess
                - analysis/zonal stats



            :param kwargs:
            :return:
            """
            logger.info(f'Assessing variable {self.name}')

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
                    sources = self.resolve(evaluate=True, **kwargs)



            return self.evaluate(**kwargs)





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




if __name__ == '__main__':
    logger = util.setup_logger(name='rapida', level=logging.INFO)
    admin_layer = '/data/adhoc/MDA/adm/adm3transn.fgb'
    from rich.progress import Progress


    with Session() as ses:

            popvars = ses.config['variables']['population']
            fk = list(popvars.keys())[0]
            fv = popvars[fk]
            d = popvars



            with Progress(disable=False) as progress:

                for var_name, var_data in d.items():
                    progress.update(task_id=assess_task, advance=1, description=f'Processing {var_name}')
                    v = Variable(name=var_name, component='population', **var_data)
                    r = v(year=2020, country='MDA', force_compute=False, admin=admin_layer, progress=progress)



                #progress.remove_task(assess_task)


