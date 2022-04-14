// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python

#include "FFSystem.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {

// specify the python module. Note that the name must explicitly match the PROJECT() name provided
// in CMakeLists (with an underscore in front)
PYBIND11_MODULE(_forward_flux, m)
    {
    detail::export_FFSystem(m);
    }

    } // end namespace hoomd
