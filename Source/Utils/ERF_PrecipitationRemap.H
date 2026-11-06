#ifndef ERF_PRECIPITATION_REMAP_H_
#define ERF_PRECIPITATION_REMAP_H_

#include <AMReX_MultiFab.H>
#include <AMReX_Periodicity.H>

namespace erf_precipitation {

// The destination already contains the coarse baseline in newly refined cells.
// Only old valid cells carry current accumulated precipitation: SLM updates do
// not refresh the old halo. Copying that halo would overwrite the coarse values
// and can also overwrite valid fine data where old boxes meet.
inline void
restore_accumulation (amrex::MultiFab& destination,
                      const amrex::MultiFab& old,
                      const amrex::Periodicity& periodicity)
{
    amrex::IntVect ngv = destination.nGrowVect();
    ngv[2] = 0;
    destination.ParallelCopy(old, 0, 0, 1, amrex::IntVect(0), ngv, periodicity);
    destination.FillBoundary(periodicity);
}

} // namespace erf_precipitation

#endif
