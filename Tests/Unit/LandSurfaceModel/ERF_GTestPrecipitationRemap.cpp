#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <gtest/gtest.h>

#include "ERF_PrecipitationRemap.H"

namespace {

using namespace amrex;

const IntVect ghosts(2, 1, 0);

MultiFab
field (const BoxArray& boxes, Real value)
{
    MultiFab result(boxes, DistributionMapping(boxes), 1, ghosts);
    result.setVal(value);
    return result;
}

void
set_accumulation (MultiFab& field)
{
    // Model the production update: valid cells evolve; ghosts retain old data.
    for (MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto values = field.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            values(i,j,k) = Real(10+i);
        });
    }
}

template<class Expected>
void
expect_values (const MultiFab& actual, Expected expected)
{
    MultiFab error(actual.boxArray(), actual.DistributionMap(), 1, 0);
    for (MFIter mfi(actual); mfi.isValid(); ++mfi) {
        const auto values = actual.const_array(mfi);
        const auto errors = error.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            errors(i,j,k) = values(i,j,k) - expected(i,j,k);
        });
    }
    EXPECT_FALSE(error.contains_nan());
    EXPECT_EQ(error.norm0(), Real(0)); // Exact copies of small integer values.
}

// Named device callables avoid extended lambdas in GoogleTest's protected
// TestBody member, which nvcc does not support.
struct CoarseAndRetained
{
    AMREX_GPU_HOST_DEVICE Real operator() (int i, int, int) const noexcept
    {
        return (i >= 2 && i <= 5) ? Real(10+i) : Real(3);
    }
};

struct Retained
{
    AMREX_GPU_HOST_DEVICE Real operator() (int i, int, int) const noexcept
    {
        return Real(10+i);
    }
};

void expect_periodic_halos (MultiFab const& current)
{
    // Include horizontal ghosts, but only y coordinates inside the physical
    // domain. Physical-boundary extrapolation is not this helper's contract.
    MultiFab error(current.boxArray(), current.DistributionMap(), 1, ghosts);
    error.setVal(Real(0));
    for (MFIter mfi(current); mfi.isValid(); ++mfi) {
        Box region = mfi.fabbox();
        region.setSmall(1, 0);
        region.setBig(1, 3);
        const auto values = current.const_array(mfi);
        const auto errors = error.array(mfi);
        ParallelFor(region, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const int wrapped_i = (i + 8) % 8;
            errors(i,j,k) = values(i,j,k) - Real(10+wrapped_i);
        });
    }
    EXPECT_FALSE(error.contains_nan(0, 1, ghosts));
    EXPECT_EQ(error.norm0(0, 1, ghosts), Real(0));
}

} // namespace

TEST(PrecipitationRemap, NewFineCellsKeepTheirCoarseBaseline)
{
    const Box old_box(IntVect(2,0,0), IntVect(5,3,0));
    const Box new_box(IntVect(0,0,0), IntVect(7,3,0));
    auto old = field(BoxArray(old_box), Real(0));
    set_accumulation(old);
    auto current = field(BoxArray(new_box), Real(3));

    erf_precipitation::restore_accumulation(current, old, Periodicity::NonPeriodic());
    expect_values(current, CoarseAndRetained{});
}

TEST(PrecipitationRemap, RepartitioningNeverUsesStaleSourceGhosts)
{
    const Box domain(IntVect(0,0,0), IntVect(7,3,0));
    BoxArray old_boxes(domain);
    old_boxes.maxSize(4);
    auto old = field(old_boxes, Real(-23));
    set_accumulation(old);
    auto current = field(BoxArray(domain), Real(3));

    erf_precipitation::restore_accumulation(current, old, Periodicity::NonPeriodic());
    expect_values(current, Retained{});
}

TEST(PrecipitationRemap, PeriodicAndInternalHalosComeFromCurrentValidData)
{
    const Box domain(IntVect(0,0,0), IntVect(7,3,0));
    const RealBox physical(0,0,0,8,4,1);
    const Geometry geom(domain, physical, CoordSys::cartesian, {1,0,0});
    auto old = field(BoxArray(domain), Real(-23));
    set_accumulation(old);
    BoxArray new_boxes(domain);
    new_boxes.maxSize(4);
    auto current = field(new_boxes, Real(3));

    erf_precipitation::restore_accumulation(current, old, geom.periodicity());

    expect_periodic_halos(current);
}
