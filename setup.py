from setuptools import setup, find_packages
def get_requirements(file_path:str)->List[str]:
    requirements =[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', "") for req in requirements]


setup(
    name = 'TEST PROJECT',
    version='0.0.1',
    author='hemanth',
    author_email='hemanth9886609@gmail.com',
    # description=
    # long_description= # type: ignore
    # url=
    packages=find_packages(),
    install_requirements=[
        'numpy',
        'pandas'
    ]
)