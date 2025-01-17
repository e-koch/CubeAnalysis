#!/usr/bin/env python

import os
import sys
from setuptools import setup


setup(use_scm_version={'write_to': os.path.join('cube_analysis', 'version.py')})
