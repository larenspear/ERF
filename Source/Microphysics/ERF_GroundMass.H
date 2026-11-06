#ifndef ERF_GROUND_MASS_H_
#define ERF_GROUND_MASS_H_

#include <AMReX_MultiFab.H>
#include <string>

// Cumulative kg per horizontal cell, stored on the physical bottom of a 3D
// atmospheric layout. These fields are independent of precipitation depths.
struct GroundMassAccumulationField
{
    std::string name;
    amrex::MultiFab* mass;
};
using GroundMassAccumulationFields = amrex::Vector<GroundMassAccumulationField>;

#endif
