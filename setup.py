from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

import sys
import os
import platform
from shutil import which
import subprocess

IS_WINDOWS = (platform.system() == 'Windows')   # get the operating system
STATE_SIZE = 8                                  # the compiled code supports MHNs with maximum size of 32 * STATE_SIZE
GENERATE_DEBUG_HTML = False                     # set this to True so that Cython generates a optimization HTML file

assert STATE_SIZE > 0                           # make sure STATE_SIZE is greater zero

with open("README.md", 'r') as f:
    long_description = f.read()


def compile_cuda_code(folder, cuda_filename, lib_name):
    """
    This function compiles the CUDA code of cuda_state_space_restriction.cu containing the CUDA function for State Space Restriction
    """
    if IS_WINDOWS:
        output_filename = os.path.join(folder, f"{lib_name}.dll")
    else:
        output_filename = os.path.join(folder, f"lib{lib_name}.so")

    cuda_filename = os.path.join(folder, cuda_filename)
    # check if the shared library file was modified after the source file
    try:
        shared_lib_latest_version = (os.path.getmtime(cuda_filename) - os.path.getmtime(output_filename) < 0)
    except FileNotFoundError:
        shared_lib_latest_version = False

    # check if compilation is even necessary or if the CUDA file was already compiled
    # for that we check if the shared library file exists and if it was modified after the source file
    # if the argument --force is given to setup.py, the CUDA code is always recompiled
    if "--force" not in sys.argv and os.path.isfile(output_filename) and shared_lib_latest_version:
        print("No need to compile CUDA code again...")
        return

    # command to compile the CUDA code using nvcc
    compile_command = ['nvcc', '-o', output_filename, '--shared', cuda_filename, f'-DSTATE_SIZE={STATE_SIZE}']
    if not IS_WINDOWS:
        compile_command += ['-Xcompiler', '-fPIC']

    # execute command and print the output
    print(subprocess.run(compile_command, stdout=subprocess.PIPE).stdout.decode('utf-8', "ignore"))


# check if nvcc (the cuda compiler) is available on the device
nvcc_available = int(which('nvcc') is not None)

libraries = []
if nvcc_available:
    libraries.append(os.path.abspath("./mhn/ssr/CudaStateSpaceRestriction"))
    libraries.append(os.path.abspath("./mhn/original/CudaFullStateSpace"))
    compile_cuda_code("./mhn/ssr/", "cuda_state_space_restriction.cu", "CudaStateSpaceRestriction")
    compile_cuda_code("./mhn/original/", "cuda_full_state_space.cu", "CudaFullStateSpace")


# define compile options for the Cython files
ext_modules = [
    Extension(
        "mhn.ssr.state_storage",
        ["./mhn/ssr/state_storage.pyx"],
        extra_compile_args=[
            f'-DSTATE_SIZE={STATE_SIZE}'
        ]
    ),
    Extension(
        "mhn.ssr.state_space_restriction",
        ["./mhn/ssr/state_space_restriction.pyx"],
        libraries=libraries,
        library_dirs=["./mhn/ssr/", ".mhn/original/"],
        runtime_library_dirs=None if IS_WINDOWS else ["./mhn/ssr/", ".mhn/original/"],
        include_dirs=['./mhn/ssr/', ".mhn/original/"],
        extra_compile_args=[
            '/Ox' if IS_WINDOWS else '-O2',
            f'-DSTATE_SIZE={STATE_SIZE}'
        ],
        extra_link_args=[]
    ),
    Extension(
        "mhn.original.Likelihood",
        ["./mhn/original/Likelihood.pyx"],
        libraries=libraries,
        library_dirs=["./mhn/original/"],
        runtime_library_dirs=None if IS_WINDOWS else ["./mhn/original/"],
        include_dirs=['./mhn/original/'],
        extra_compile_args=[
            '/Ox' if IS_WINDOWS else '-O2'
        ]
    ),
    Extension(
        "mhn.original.PerformanceCriticalCode",
        ["./mhn/original/PerformanceCriticalCode.pyx"],
        extra_compile_args=[
            '/Ox' if IS_WINDOWS else '-O2',
            '/openmp' if IS_WINDOWS else '-fopenmp',
            f'-DNUMBER_OF_THREADS={os.cpu_count()}'
        ]
    ),
    Extension(
        "mhn.original.ModelConstruction",
        ["./mhn/original/ModelConstruction.pyx"],
        extra_compile_args=[
            '/Ox' if IS_WINDOWS else '-O2'
        ]
    )
]

setup(
    name="mhn",
    version="0.0.5",
    packages=find_packages(),
    author="Stefan Vocht",
    description="Contains functions to train and work with Mutual Hazard Networks",
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=cythonize(ext_modules,
                          annotate=GENERATE_DEBUG_HTML,
                          compile_time_env=dict(
                                                NVCC_AVAILABLE=nvcc_available,
                                                STATE_SIZE=STATE_SIZE
                                                )
                          ),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.1.0'
    ]
)
