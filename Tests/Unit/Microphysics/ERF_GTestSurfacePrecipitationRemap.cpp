#include <gtest/gtest.h>

#include "ERF_Kessler.H"
#include "ERF_PrecipitationRemap.H"
#include "ERF_SurfacePrecipitationRemap.H"

namespace {

using namespace amrex;

Geometry geometry (const Box& domain, bool periodic_x = false)
{
    return Geometry(domain, RealBox(0,0,0,8,4,4), CoordSys::cartesian,
                    {periodic_x,0,0});
}

MultiFab native_field (const BoxArray& boxes, Real initial)
{
    MultiFab result(boxes, DistributionMapping(boxes), 1, IntVect(1));
    result.setVal(initial);
    return result;
}

void set_surface (MultiFab& field, const Geometry& geom, Real value, Real x_slope = 0)
{
    const int bottom = geom.Domain().smallEnd(2);
    for (MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto values = field.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (k == bottom) { values(i,j,k) = value + x_slope*Real(i); }
        });
    }
    Gpu::streamSynchronize();
}

struct SurfaceContract
{
    int bottom;
    int retained_low;
    int retained_high;
    Real retained_value;
    Real coarse_value;
    Real upper_value;
    AMREX_GPU_HOST_DEVICE Real operator() (int i, int, int k) const noexcept
    {
        if (k != bottom) { return upper_value; }
        return i >= retained_low && i <= retained_high ? retained_value : coarse_value;
    }
};

template<class Expected>
void expect_values (const MultiFab& field, Expected expected)
{
    MultiFab difference(field.boxArray(), field.DistributionMap(), 1, 0);
    for (MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto values = field.const_array(mfi);
        const auto error = difference.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            error(i,j,k) = values(i,j,k) - expected(i,j,k);
        });
    }
    EXPECT_FALSE(difference.contains_nan());
    EXPECT_EQ(difference.norm0(), Real(0));
}

void expect_same_surface (const MultiFab& a, const MultiFab& b, const Geometry& geom)
{
    auto left = erf_precipitation::capture_surface_accumulation(a, geom);
    auto right = erf_precipitation::capture_surface_accumulation(b, geom);
    ASSERT_TRUE(left);
    ASSERT_TRUE(right);
    MultiFab::Subtract(*left, *right, 0, 0, 1, 0);
    EXPECT_EQ(left->norm0(), Real(0));
}

} // namespace

TEST(SurfacePrecipitationRemap, SelectsPhysicalBottomAcrossVerticalDecomposition)
{
    const Box domain(IntVect(0,0,5), IntVect(7,3,12));
    const auto geom = geometry(domain);
    BoxArray boxes(domain);
    boxes.maxSize(IntVect(4,4,4));
    auto native = native_field(boxes, Real(-991));
    set_surface(native, geom, Real(12));
    auto surface = erf_precipitation::capture_surface_accumulation(native, geom);
    ASSERT_TRUE(surface);
    EXPECT_TRUE(surface->boxArray().isDisjoint());
    EXPECT_EQ(surface->boxArray().minimalBox(), Box(IntVect(0), IntVect(7,3,0)));
    EXPECT_EQ(surface->min(0), Real(12));
    EXPECT_EQ(surface->max(0), Real(12));
    EXPECT_EQ(surface->nGrowVect(), IntVect(0));
}

TEST(SurfacePrecipitationRemap, RetainedAndNewCellsSurviveMovingAndVerticalRefinement)
{
    const Box coarse_domain(IntVect(0), IntVect(7,3,3));
    const Box fine_domain(IntVect(0), IntVect(15,7,7));
    const auto coarse_geom = geometry(coarse_domain);
    const auto fine_geom = geometry(fine_domain);
    BoxArray coarse_boxes(coarse_domain);
    coarse_boxes.maxSize(IntVect(4,4,2));
    auto coarse = native_field(coarse_boxes, Real(-999));
    set_surface(coarse, coarse_geom, Real(7));
    BoxArray old_boxes(Box(IntVect(4,0,0), IntVect(11,7,7)));
    old_boxes.maxSize(IntVect(4,4,4));
    auto old = native_field(old_boxes, Real(-123));
    set_surface(old, fine_geom, Real(12));
    auto retained = erf_precipitation::capture_surface_accumulation(old, fine_geom);
    BoxArray new_boxes(Box(IntVect(8,0,0), IntVect(15,7,7)));
    new_boxes.maxSize(IntVect(4,4,4));
    auto current = native_field(new_boxes, Real(-17));
    erf_precipitation::restore_native_surface_accumulation(
        current, retained.get(), fine_geom, &coarse, &coarse_geom, IntVect(2));
    expect_values(current, SurfaceContract{0,8,11,Real(12),Real(7),Real(-17)});
}

TEST(SurfacePrecipitationRemap, RebuildsPoisonedPeriodicAndPhysicalCoarseHalos)
{
    const Box coarse_domain(IntVect(0), IntVect(7,3,3));
    const Box fine_domain(IntVect(0), IntVect(15,7,7));
    const auto coarse_geom = geometry(coarse_domain, true);
    const auto fine_geom = geometry(fine_domain, true);
    BoxArray coarse_boxes(coarse_domain);
    coarse_boxes.maxSize(IntVect(4,4,4));
    auto coarse_a = native_field(coarse_boxes, Real(-999));
    auto coarse_b = native_field(coarse_boxes, Real(888));
    set_surface(coarse_a, coarse_geom, Real(10), Real(1));
    set_surface(coarse_b, coarse_geom, Real(10), Real(1));
    auto current_a = native_field(BoxArray(fine_domain), Real(0));
    auto current_b = native_field(BoxArray(fine_domain), Real(0));
    erf_precipitation::restore_native_surface_accumulation(
        current_a, nullptr, fine_geom, &coarse_a, &coarse_geom, IntVect(2));
    erf_precipitation::restore_native_surface_accumulation(
        current_b, nullptr, fine_geom, &coarse_b, &coarse_geom, IntVect(2,2,4));
    expect_same_surface(current_a, current_b, fine_geom);
    auto fine = erf_precipitation::capture_surface_accumulation(current_a, fine_geom);
    ASSERT_TRUE(fine);
    // Linear conservative interpolation preserves each coarse-cell mean;
    // vertical refinement must not change a surface integral.
    EXPECT_EQ(fine->sum(0) / Real(4), Real(432)); // sum_{i,j} (10+i)
    EXPECT_GE(fine->min(0), Real(10));
    EXPECT_LE(fine->max(0), Real(17));
}

TEST(SurfacePrecipitationRemap, EmptyBottomCoverageLeavesAtmosphericValuesAlone)
{
    const Box domain(IntVect(0), IntVect(7,3,7));
    const auto geom = geometry(domain);
    auto field = native_field(BoxArray(Box(IntVect(0,0,4), IntVect(7,3,7))), Real(9));
    EXPECT_FALSE(erf_precipitation::capture_surface_accumulation(field, geom));
    erf_precipitation::restore_native_surface_accumulation(field, nullptr, geom);
    EXPECT_EQ(field.min(0), Real(9));
    EXPECT_EQ(field.max(0), Real(9));
}

TEST(SurfacePrecipitationRemap, NativeMembershipAndSlmUnitsKeepFrozenSpeciesSeparate)
{
    const Box domain(IntVect(0), IntVect(7,3,3));
    const auto geom = geometry(domain);
    auto rain = native_field(BoxArray(domain), Real(0));
    auto snow = native_field(BoxArray(domain), Real(0));
    auto unrelated = native_field(BoxArray(domain), Real(99));
    set_surface(rain, geom, Real(6));
    set_surface(snow, geom, Real(4));
    SurfacePrecipAccumulationSources sources;
    sources.rain = {&rain, Real(2)};
    sources.snow = {&snow, Real(3)};
    EXPECT_TRUE(erf_precipitation::is_native_surface_accumulator(&rain, sources));
    EXPECT_TRUE(erf_precipitation::is_native_surface_accumulator(&snow, sources));
    EXPECT_FALSE(erf_precipitation::is_native_surface_accumulator(&unrelated, sources));
    EXPECT_FALSE(erf_precipitation::is_native_surface_accumulator(nullptr, sources));
    auto rain_saved = erf_precipitation::capture_surface_accumulation(rain, geom);
    auto snow_saved = erf_precipitation::capture_surface_accumulation(snow, geom);
    rain.setVal(Real(0));
    snow.setVal(Real(0));
    erf_precipitation::restore_native_surface_accumulation(rain, rain_saved.get(), geom);
    erf_precipitation::restore_native_surface_accumulation(snow, snow_saved.get(), geom);
    expect_values(rain, SurfaceContract{0,0,7,Real(6),Real(0),Real(0)});
    expect_values(snow, SurfaceContract{0,0,7,Real(4),Real(0),Real(0)});
    auto baseline = native_field(BoxArray(Box(IntVect(0), IntVect(7,3,0))), Real(-1));
    erf_precipitation::seed_slm_baseline_from_surface(baseline, sources, geom);
    EXPECT_EQ(baseline.min(0), Real(12)); // rain only: 6 * 2
    EXPECT_EQ(baseline.max(0), Real(12));
    sources.total = {&snow, Real(5)};
    erf_precipitation::seed_slm_baseline_from_surface(baseline, sources, geom);
    EXPECT_EQ(baseline.min(0), Real(20)); // total takes precedence
    EXPECT_EQ(baseline.max(0), Real(20));
}

TEST(SurfacePrecipitationRemap, ActualKesslerReinitializationPreservesFirstAndLaterRates)
{
    const Box domain(IntVect(0), IntVect(7,3,3));
    const auto geom = geometry(domain);
    const BoxArray boxes(domain);
    const DistributionMapping dm(boxes);
    MultiFab state(boxes, dm, 5, 1);
    state.setVal(Real(0));
    std::unique_ptr<MultiFab> zphys, detj;
    SolverChoice sc;
    sc.moisture_type = MoistureType::Kessler;
    Kessler scheme;
    scheme.Define(sc);
    scheme.Init(state, boxes, geom, 0, zphys, detj);
    auto* accumulator = scheme.Qmoist_Ptr(0);
    set_surface(*accumulator, geom, Real(12));
    auto saved = erf_precipitation::capture_surface_accumulation(*accumulator, geom);
    scheme.Define(sc);
    scheme.Init(state, boxes, geom, 0, zphys, detj);
    accumulator = scheme.Qmoist_Ptr(0);
    EXPECT_EQ(accumulator->max(0), Real(0)); // actual initializer clears history
    erf_precipitation::restore_native_surface_accumulation(*accumulator, saved.get(), geom);
    accumulator->plus(Real(2), 0, 1, 0);
    auto now = erf_precipitation::capture_surface_accumulation(*accumulator, geom);
    ASSERT_TRUE(now);
    EXPECT_EQ(now->min(0) - Real(12), Real(2));
    const Real baseline = now->min(0);
    accumulator->plus(Real(3), 0, 1, 0);
    now = erf_precipitation::capture_surface_accumulation(*accumulator, geom);
    EXPECT_EQ(now->min(0) - baseline, Real(3));
}

TEST(SurfacePrecipitationRemap, SlmSeedUsesNewNativeUnitsAndKeepsOldIntervalHistory)
{
    const Box domain(IntVect(0), IntVect(15,7,7));
    const auto geom = geometry(domain);
    auto native = native_field(BoxArray(domain), Real(0));
    set_surface(native, geom, Real(7));
    BoxArray old_surface_boxes(Box(IntVect(4,0,0), IntVect(11,7,0)));
    auto old_native = native_field(old_surface_boxes, Real(-999));
    old_native.setVal(Real(6), 0, 1, 0);
    native.ParallelCopy(old_native, 0, 0, 1, 0, 0);
    auto old_baseline = native_field(old_surface_boxes, Real(-888));
    old_baseline.setVal(Real(11), 0, 1, 0);
    BoxArray new_surface_boxes(Box(IntVect(8,0,0), IntVect(15,7,0)));
    new_surface_boxes.maxSize(IntVect(4,4,1));
    auto baseline = native_field(new_surface_boxes, Real(-333));
    SurfacePrecipAccumulationSources sources;
    sources.rain = {&native, Real(2)};
    erf_precipitation::seed_slm_baseline_from_surface(baseline, sources, geom);
    erf_precipitation::restore_accumulation(baseline, old_baseline, geom.periodicity());
    // The old baseline (11) intentionally differs from the old native total
    // in SLM units (6*2). Regridding must keep that unconsumed interval history.
    expect_values(baseline, SurfaceContract{0,8,11,Real(11),Real(14),Real(0)});
}
