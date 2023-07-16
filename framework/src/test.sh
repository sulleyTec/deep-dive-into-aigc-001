#!/usr/bin/bash

#pytest test.py::test_mm
#pytest test.py::test_permute
#pytest test.py::test_mean
#pytest test.py::test_var
#pytest test.py::test_broadcast
pytest test.py::test_element_wise_mul
pytest test.py::test_element_wise_add

