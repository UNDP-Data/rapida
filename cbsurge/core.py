import os.path
import asyncio
from sympy.parsing.sympy_parser import parse_expr
from pydantic import BaseModel, FilePath
from typing import Optional, List, Union
import re

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
    for p in walk_packages():
        if p.name.startswith(root_package_name) and p.name.endswith(variables_module) and not p.ispkg:
            m = importlib.import_module(name=p.name)
            if hasattr(m, function_name):
                var_dict = getattr(m, 'generate_variables')()
                component_name = p.name.split('.')[-2]

                vars_dict[variables_module][component_name] = var_dict
    return vars_dict

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
        self.__dict__.update(kwargs)

        try:
            parsed_expr = parse_expr(self.sources)
            self.variables = [s.name for s in parsed_expr.free_symbols]

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

    def download(self, **kwargs):
        src_path = self.interpolate_source(template=self.source, **self._update_(**kwargs))
        if os.path.exists(self.local_path): return self.local_path
        downloaded_file = asyncio.run(blobstorage.download_blob(src_path=src_path,dst_path=self.local_path)                             )
        assert downloaded_file == self.local_path, f'The local_path differs from {downloaded_file}'
        return downloaded_file


    def assess(self, **kwargs):
        args = self._update_(**kwargs)
        with Session() as s:
            src_rasters = list()
            vars_ops = list()
            for var in self.variables:
                var_dict = s.get_variable(component=self.component, variable=var)
                v = self.__class__(name=var, component=self.component, **var_dict.updatre(**args))
                p = v(**args) # call & resolve
                logger.info(f'{var} was resolved to {p}')
                src_rasters.append(p)
                vars_ops.append((var, 'sum'))

            gdf = zonal_stats(src_rasters=src_rasters,src_vector=kwargs['admin'], vars_ops=vars_ops)
            expr = f'{self.name}={self.sources}'
            gdf.eval(expr, inplace=True)
            gdf.to_file(f'/tmp/popstats.fgb', driver='FlatGeobuf',index=False,engine='pyogrio')

    def __call__(self, force_compute=False, **kwargs):
            args = self._update_(**kwargs)
            if self.source:

                if not os.path.exists(self._source_folder_):
                    os.makedirs(self._source_folder_)
                src_path = self.interpolate_source(template=self.source, **args)
                _, file_name = os.path.split(src_path)
                self.local_path = os.path.join(self._source_folder_, file_name)
                if os.path.exists(self.local_path):
                    #os.remove(self.local_path)
                    return self.local_path


            if self.source and not force_compute:
                return self.download(**args)
            # use sources to compute
            if self.sources:
                if self.variables:
                    logger.info(f'Going to compute {self.name} from {self.sources}')
                    assert 'admin' in kwargs, f'Admin layer is required to compute zonal stats'
                    return self.assess(**args)

                else:
                    logger.info(f'Going to sum up {self.name} from source files')\
                    # interpolate templates
                    source_blobs = list()
                    for source_template in self.sources:
                        source_file_path = self.interpolate_source(template=source_template, **kwargs)
                        source_blobs.append(source_file_path)
                    downloaded_files = asyncio.run(
                        blobstorage.download_blobs(src_blobs=source_blobs, dst_folder=self._source_folder_,
                                                   progress=kwargs['progress'])
                    )
                    assert len(self.sources) == len(downloaded_files), f'Not all sources were downloaded for {self.name} variable'
                    sumup(src_rasters=downloaded_files,dst_raster=self.local_path)
                    logger.info(f'{self.local_path} was created for variable {self.name}')




    def interpolate_source(self, template=None, **kwargs):
        """
        Resolve file paths with the provided kwargs, ensuring that the necessary variables are included.
        """

        kwargs['country_lower'] = self.country.lower()
        template_vars = set(re.findall(self._extractor_, template))
        for template_var in template_vars:
            if not template_var in kwargs:
                assert hasattr(self,template_var), f'"{template_var}"  is required to generate source files'

        return template.format(**kwargs)




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
                assess_task = progress.add_task(
                    description=f'[red]Going to process {len(d)} variables', total=len(d))
                for var_name, var_data in d.items():
                    progress.update(task_id=assess_task, advance=1, description=f'Processing {var_name}')
                    v = SurgeVariable(name=var_name, component='population', **var_data)
                    r = v(year=2020, country='MDA', force_compute=False, admin=admin_layer, progress=progress)



                #progress.remove_task(assess_task)


