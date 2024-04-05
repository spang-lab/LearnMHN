from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

import sys
import os
import platform
from shutil import which
import subprocess

VERSION = "0.0.19"                                 # current package version

IS_WINDOWS = (platform.system() == 'Windows')      # get the operating system
STATE_SIZE = 8                                     # the compiled code supports MHNs with maximum size of 32 * STATE_SIZE
GENERATE_DEBUG_HTML = False                        # set this to True so that Cython generates an optimization HTML file
NO_CUDA_INSTALLATION_FLAG = "INSTALL_MHN_NO_CUDA"  # set this environmental variable to install CPU version only

assert STATE_SIZE > 0                              # make sure STATE_SIZE is greater zero

with open("README.md", 'r') as f:
    long_description = f.read()


def create_metadata_file(**metadata):
    """
    This function creates the METADATA file which contains metadata about this package that can be accessed at runtime.
    """
    with open("mhn/METADATA", "w") as file:
        for key in metadata:
            file.write(f"{key} {metadata[key]}\n")


def compile_cuda_code(folder, cuda_filename, lib_name, *extra_compile_args, additional_cuda_files=None):
    """
    This function compiles the CUDA code of this package to run functions on the GPU
    """
    if additional_cuda_files is None:
        additional_cuda_files = []
    if IS_WINDOWS:
        output_filename = os.path.join(folder, f"{lib_name}.dll")
    else:
        output_filename = os.path.join(folder, f"lib{lib_name}.so")

    cuda_filename = os.path.join(folder, cuda_filename)
    # check if the shared library file was modified after the source files
    shared_lib_latest_version = True
    try:
        for cuda_file in [cuda_filename] + additional_cuda_files:
            shared_lib_latest_version &= (os.path.getmtime(cuda_file) - os.path.getmtime(output_filename) < 0)
    except FileNotFoundError:
        shared_lib_latest_version = False

    # check if compilation is even necessary or if the CUDA file was already compiled
    # for that we check if the shared library file exists and if it was modified after the source file
    # if the argument --force is given to setup.py, the CUDA code is always recompiled
    if "--force" not in sys.argv and os.path.isfile(output_filename) and shared_lib_latest_version:
        print("No need to compile CUDA code again...")
        return

    # command to compile the CUDA code using nvcc
    compile_command = ['nvcc', '-o', output_filename, '--shared', cuda_filename, *additional_cuda_files,
                       f'-DSTATE_SIZE={STATE_SIZE}', *extra_compile_args]
    if not IS_WINDOWS:
        compile_command += ['-Xcompiler', '-fPIC']

    # execute command and print the output
    print(subprocess.run(compile_command, stdout=subprocess.PIPE).stdout.decode('utf-8', "ignore"))


# check if nvcc (the cuda compiler) is available on the device
nvcc_available = int(which('nvcc') is not None)

# check if manual instruction not to use CUDA was given
if NO_CUDA_INSTALLATION_FLAG in os.environ:
    nvcc_available = 0

libraries = []
extra_cuda_link_args = []
# only compile CUDA code if nvcc is available and if we do not create a source distribution
if nvcc_available and 'sdist' not in sys.argv:
    libraries.append("CudaStateSpaceRestriction")
    libraries.append("CudaFullStateSpace")
    compile_cuda_code("./mhn/original/", "cuda_full_state_space.cu", "CudaFullStateSpace",
                      additional_cuda_files=["./mhn/original/cuda_inverse_by_substitution.cu"])
    compile_cuda_code("./mhn/ssr/", "cuda_state_space_restriction.cu", "CudaStateSpaceRestriction", f'-I./mhn/original/',
                      additional_cuda_files=["./mhn/original/cuda_inverse_by_substitution.cu"])
    if not IS_WINDOWS:
        extra_cuda_link_args = [
            '-Wl,-rpath,$ORIGIN/../original/',
            '-Wl,-rpath,$ORIGIN/../ssr/',
        ]


# define compile options for the Cython files
ext_modules = [
    Extension(
        "mhn.ssr.state_containers",
        ["./mhn/ssr/state_containers.pyx"],
        extra_compile_args=[
            f'-DSTATE_SIZE={STATE_SIZE}'
        ]
    ),
    Extension(
        "mhn.ssr.state_space_restriction",
        ["./mhn/ssr/state_space_restriction.pyx"],
        libraries=libraries,
        library_dirs=["./mhn/ssr/", "./mhn/original/"],
        include_dirs=['./mhn/ssr/', "./mhn/original/"],
        extra_compile_args=[
            '/Ox' if IS_WINDOWS else '-O2',
            f'-DSTATE_SIZE={STATE_SIZE}'
        ],
        extra_link_args=extra_cuda_link_args
    ),
    Extension(
        "mhn.ssr.matrix_exponential",
        ["./mhn/ssr/matrix_exponential.pyx"],
        libraries=libraries,
        library_dirs=["./mhn/ssr/", "./mhn/original/"],
        # runtime_library_dirs=None if IS_WINDOWS else ["./mhn/ssr/", "./mhn/original/"],
        include_dirs=['./mhn/ssr/', "./mhn/original/"],
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
        library_dirs=["./mhn/ssr/", "./mhn/original/"],
        include_dirs=['./mhn/ssr/', "./mhn/original/"],
        extra_compile_args=[
            '/Ox' if IS_WINDOWS else '-O2'
        ],
        extra_link_args=extra_cuda_link_args
    ),
    Extension(
        "mhn.original.PerformanceCriticalCode",
        ["./mhn/original/PerformanceCriticalCode.pyx"],
        extra_compile_args=[
            '/Ox' if IS_WINDOWS else '-O2',
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

# we only want the source code in a source distribution
if 'sdist' in sys.argv:
    ext_modules = []
else:
    create_metadata_file(version=VERSION)

setup(
    name="mhn",
    version=VERSION,
    packages=find_packages(),
    author="Stefan Vocht, Kevin Rupp, Y. Linda Hu",
    description="A package to train and work with Mutual Hazard Networks",
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=cythonize(ext_modules,
                          annotate=GENERATE_DEBUG_HTML,
                          compile_time_env=dict(
                                                NVCC_AVAILABLE=nvcc_available,
                                                STATE_SIZE=STATE_SIZE
                                                ),
                          compiler_directives={'embedsignature': True}
                          ),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
    install_requires=[
        'numpy>=1.23.0',
        'scipy>=1.10.0',
        'pandas>=1.5.3',
        'tqdm>=4.65.0',
        'matplotlib>=3.5.0'
    ],
    python_requires='>=3.8, <3.12'
)
