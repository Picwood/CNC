import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


if os.name == "nt":
    compile_args = ["/O2"]
else:
    compile_args = ["-O3"]

ext_modules = [
    Pybind11Extension(
        "cnc_toolpath_accel",
        ["src/cnc_toolpath_accel.cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
    )
]

setup(
    name="cnc-polar-engraving-accel",
    version="0.1.0",
    description="Accelerated toolpath core for planar sinusoidal polar engraving",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
