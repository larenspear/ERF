#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <gtest/gtest.h>

#include "ERF_TerrainMetrics.H"

// Motivation: erf.initial_dz without erf.grid_stretching_ratio asks init_zlevels to solve for
// the ratio, so that an idealized or metgrid case can name its first layer and its domain top
// and get a grid that closes on both.  An explicit ratio must keep building the grid it always
// has, whatever top that reaches.

using namespace amrex;

namespace {

constexpr int nz = 32;

Geometry
make_geom (Real zlo, Real zhi)
{
    const Box domain(IntVect(0,0,0), IntVect(7,7,nz-1));
    const RealBox rb(Real(0.), Real(0.), zlo, Real(800.), Real(800.), zhi);
    return Geometry(domain, rb, CoordSys::cartesian, {1, 1, 0});
}

struct ZLevels
{
    Vector<Real> zlevels;
    Vector<Real> dz;
};

ZLevels
build (Real zhi, Real ratio, Real zsurf, Real dz0)
{
    Vector<Geometry> geom{make_geom(zsurf, zhi)};
    Vector<IntVect>  ref_ratio;

    Vector<Vector<Real>> zlevels_stag(1);
    Vector<Vector<Real>> stretched_dz_h(1);
    Vector<Gpu::DeviceVector<Real>> stretched_dz_d(1);

    init_zlevels(zlevels_stag, stretched_dz_h, stretched_dz_d, geom, ref_ratio,
                 ratio, zsurf, dz0);

    return {zlevels_stag[0], stretched_dz_h[0]};
}

Real
allowance (Real scale)
{
    return Real(16.0) * nz * std::numeric_limits<Real>::epsilon() * std::abs(scale);
}

struct ThrowOnError
{
    bool previous = amrex::system::throw_exception;
    ThrowOnError () { amrex::system::throw_exception = true; }
    ~ThrowOnError () { amrex::system::throw_exception = previous; }
};

struct SuppliedNodes
{
    Vector<Real> previous;
    bool had_previous;
    explicit SuppliedNodes (const Vector<Real>& nodes)
    {
        ParmParse pp("erf");
        had_previous = pp.queryarr("terrain_z_levels", previous);
        pp.addarr("terrain_z_levels", nodes);
    }
    ~SuppliedNodes ()
    {
        ParmParse pp("erf");
        pp.remove("terrain_z_levels");
        if (had_previous) { pp.addarr("terrain_z_levels", previous); }
    }
};

struct GridStorage
{
    Vector<Vector<Real>> nodes{1};
    Vector<Vector<Real>> host{1};
    Vector<Gpu::DeviceVector<Real>> device{1};

    GridStorage ()
    {
        const Vector<Real> sentinel{Real(17)};
        device[0].resize(1);
        Gpu::copy(Gpu::hostToDevice, sentinel.begin(), sentinel.end(), device[0].begin());
    }

    Vector<Real> copied () const
    {
        Vector<Real> result(device[0].size());
        Gpu::copy(Gpu::deviceToHost, device[0].begin(), device[0].end(), result.begin());
        return result;
    }

    void init (int layers, Real top, Real ratio, Real surface, Real first)
    {
        const Box domain(IntVect(0), IntVect(3,3,layers-1));
        const RealBox rb(Real(0), Real(0), Real(0), Real(4), Real(4), top);
        const Vector<Geometry> geom{Geometry(domain, rb, CoordSys::cartesian, {1,1,0})};
        init_zlevels(nodes, host, device, geom, {}, ratio, surface, first);
    }
};

} // namespace

// Motivation: the solved ratio has to close the column on prob_hi, start one initial_dz above
// the surface, and give back the ratio an explicit run of the same grid would have used.
TEST(InitZLevels, SolvesTheRatioThatClosesOnProbHi)
{
    // 32 layers from 19.503737114205 m growing by 1.03 sum to 1024 m
    const Real dz0  = Real(19.503737114205);
    const Real ztop = Real(1024.0);
    const auto z = build(ztop, Real(0.0), Real(0.0), dz0);

    ASSERT_EQ(static_cast<int>(z.zlevels.size()), nz+1);
    ASSERT_EQ(static_cast<int>(z.dz.size()), nz);
    EXPECT_EQ(z.zlevels[0], Real(0.0));
    EXPECT_NEAR(z.zlevels[nz], ztop, allowance(ztop));
    EXPECT_NEAR(z.dz[0], dz0, allowance(ztop));

    for (int k(0); k<nz; ++k) {
        EXPECT_GT(z.dz[k], Real(0.0)) << "at k = " << k;
        EXPECT_NEAR(z.dz[k], z.zlevels[k+1] - z.zlevels[k], allowance(ztop)) << "at k = " << k;
    }
    // The top layer is assigned z_top exactly rather than by accumulation
    for (int k(1); k<nz-1; ++k) {
        // Subtracting stored nodes and translating by zsurface each introduce
        // coordinate-scale rounding. Propagate those errors through the ratio.
        const Real ratio_error = Real(8) * std::numeric_limits<Real>::epsilon() *
            std::max(std::abs(z.zlevels[k+1]), std::abs(z.zlevels[k-1])) / z.dz[k-1];
        EXPECT_NEAR(z.dz[k]/z.dz[k-1], Real(1.03),
                    std::max(Real(1.e-6), ratio_error)) << "at k = " << k;
    }
}

// Motivation: the grid is built above erf.zsurface, so the column it has to fill is
// prob_hi - zsurface rather than prob_hi.
TEST(InitZLevels, SolvedRatioStartsAtZsurface)
{
    const Real zsurf = Real(100.0);
    const Real ztop  = Real(1124.0);
    const auto z = build(ztop, Real(0.0), zsurf, Real(19.503737114205));

    EXPECT_EQ(z.zlevels[0], zsurf);
    EXPECT_NEAR(z.zlevels[nz], ztop, allowance(ztop));
    for (int k(1); k<nz-1; ++k) {
        // Subtracting stored nodes and translating by zsurface each introduce
        // coordinate-scale rounding. Propagate those errors through the ratio.
        const Real ratio_error = Real(8) * std::numeric_limits<Real>::epsilon() *
            std::max(std::abs(z.zlevels[k+1]), std::abs(z.zlevels[k-1])) / z.dz[k-1];
        EXPECT_NEAR(z.dz[k]/z.dz[k-1], Real(1.03),
                    std::max(Real(1.e-6), ratio_error)) << "at k = " << k;
    }
}

// Motivation: an explicit ratio is a request for that grid, not for one that closes on
// prob_hi, so it must still end wherever its layers do.
TEST(InitZLevels, AnExplicitRatioIsNotSolvedFor)
{
    const Real dz0  = Real(20.0);
    const auto z = build(Real(1024.0), Real(1.03), Real(0.0), dz0);

    Real top = Real(0.0);
    Real dz  = dz0;
    for (int k(0); k<nz; ++k) { top += dz; dz *= Real(1.03); }

    EXPECT_NEAR(z.zlevels[nz], top, allowance(top));
    EXPECT_GT(z.zlevels[nz], Real(1024.0));
}

// Motivation: without initial_dz or a ratio the grid stays the uniform one from the geometry.
TEST(InitZLevels, NeitherGivenIsUniform)
{
    const auto z = build(Real(1024.0), Real(0.0), Real(0.0), Real(0.0));
    for (int k(0); k<nz; ++k) {
        EXPECT_NEAR(z.dz[k], Real(32.0), allowance(Real(1024.0))) << "at k = " << k;
    }
}

// The old absolute tolerance accepted ratio 1.03 for this tiny column and
// inverted its final layer. The bounded solve must now produce positive spacing.
TEST(InitZLevels, BuildsPositiveTinyGeometricGrid)
{
    ThrowOnError errors;
#ifdef AMREX_USE_FLOAT
    const Real first = Real(1.e-5);
#else
    const Real first = Real(1.e-9);
#endif
    GridStorage grid;
    grid.init(10, Real(10.1)*first, Real(0), Real(0), first);
    ASSERT_EQ(grid.nodes[0].size(), 11);
    EXPECT_EQ(grid.nodes[0].back(), Real(10.1)*first);
    EXPECT_EQ(grid.copied(), grid.host[0]);
    for (const Real height : grid.host[0]) { EXPECT_GT(height, Real(0)); }
}

TEST(InitZLevels, RejectsInvalidSuppliedNodesBeforeDeviceCopy)
{
    ThrowOnError errors;
    const Real inf = std::numeric_limits<Real>::infinity();
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    for (const Real bad : {Real(1), Real(0), inf, -inf, nan}) {
        SCOPED_TRACE(bad);
        SuppliedNodes supplied({Real(0), Real(1), bad, Real(3)});
        GridStorage grid;
        EXPECT_THROW(grid.init(3, Real(3), Real(0), Real(0), Real(0)),
                     std::runtime_error);
        EXPECT_EQ(grid.copied(), Vector<Real>{Real(17)});
    }
}

TEST(InitZLevels, RejectsCollapsedAndOverflowedExplicitGrids)
{
    ThrowOnError errors;
    // Positive requested thickness can round away when added to the surface.
    GridStorage collapsed;
    EXPECT_THROW(collapsed.init(3, Real(3), Real(1), Real(1.e20), Real(1)),
                 std::runtime_error);
    EXPECT_EQ(collapsed.copied(), Vector<Real>{Real(17)});

    GridStorage overflowed;
    EXPECT_THROW(overflowed.init(3, Real(3), Real(4), Real(0),
                                std::numeric_limits<Real>::max()/Real(4)),
                 std::runtime_error);
    EXPECT_EQ(overflowed.copied(), Vector<Real>{Real(17)});
}

TEST(InitZLevels, RefreshRejectsInvalidNodesBeforeDeviceCopy)
{
    ThrowOnError errors;
    const Real inf = std::numeric_limits<Real>::infinity();
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real largest = std::numeric_limits<Real>::max();
    const Vector<Vector<Real>> cases{
        {}, {Real(0)}, {Real(0), Real(0)}, {Real(1), Real(0)},
        {nan, Real(1)}, {Real(0), nan}, {Real(0), inf}, {-inf, Real(0)},
        {-largest, largest}}; // Finite, increasing nodes whose difference overflows.
    for (const auto& nodes : cases) {
        GridStorage grid;
        grid.nodes[0] = nodes;
        EXPECT_THROW(update_stretched_dz(0, grid.nodes, grid.host, grid.device),
                     std::runtime_error);
        EXPECT_EQ(grid.copied(), Vector<Real>{Real(17)});
    }
}

TEST(InitZLevels, RefreshRejectsNegativeLevelBeforeIndexing)
{
    ThrowOnError errors;
    GridStorage grid;
    grid.nodes[0] = {Real(0), Real(1)};
    EXPECT_THROW(update_stretched_dz(-1, grid.nodes, grid.host, grid.device),
                 std::runtime_error);
    EXPECT_EQ(grid.copied(), Vector<Real>{Real(17)});
}

TEST(InitZLevels, CopiesValidSuppliedAndRefinedHeightsExactly)
{
    SuppliedNodes supplied({Real(0), Real(1), Real(3), Real(6)});
    const RealBox rb(Real(0), Real(0), Real(0), Real(4), Real(4), Real(6));
    const Vector<Geometry> geom{
        Geometry(Box(IntVect(0), IntVect(3,3,2)), rb, CoordSys::cartesian, {1,1,0}),
        Geometry(Box(IntVect(0), IntVect(7,7,5)), rb, CoordSys::cartesian, {1,1,0})};
    Vector<Vector<Real>> nodes(2), host(2);
    Vector<Gpu::DeviceVector<Real>> device(2);
    init_zlevels(nodes, host, device, geom, {IntVect(2)}, Real(0), Real(0), Real(0));
    EXPECT_EQ(host[0], (Vector<Real>{Real(1), Real(2), Real(3)}));
    EXPECT_EQ(host[1], (Vector<Real>{Real(0.5), Real(0.5), Real(1), Real(1), Real(1.5), Real(1.5)}));
    for (int lev = 0; lev < 2; ++lev) {
        Vector<Real> copied(device[lev].size());
        Gpu::copy(Gpu::deviceToHost, device[lev].begin(), device[lev].end(), copied.begin());
        EXPECT_EQ(copied, host[lev]);
    }
    nodes[1] = {Real(0), Real(1), Real(2), Real(4), Real(6), Real(9), Real(12)};
    update_stretched_dz(1, nodes, host, device);
    Vector<Real> copied(device[1].size());
    Gpu::copy(Gpu::deviceToHost, device[1].begin(), device[1].end(), copied.begin());
    EXPECT_EQ(copied, (Vector<Real>{Real(1), Real(1), Real(2), Real(2), Real(3), Real(3)}));
    EXPECT_EQ(host[0], (Vector<Real>{Real(1), Real(2), Real(3)}));
}
