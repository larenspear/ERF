#include <gtest/gtest.h>
#include <limits>

#include "ERF_PrecipitationHistory.H"

namespace {
using namespace amrex;

MultiFab history_field (const BoxArray& boxes, Real value, int ghosts = 0)
{
    MultiFab field(boxes, DistributionMapping(boxes), 1, ghosts);
    field.setVal(value);
    return field;
}

Box history_box () { return Box(IntVect(0),IntVect(7,5,3)); }
}

TEST(PrecipitationHistory, RetainsPendingIntervalsForBothShiftSigns)
{
    const BoxArray boxes(history_box());
    for (Real current : {Real(2),Real(14)}) {
        for (Real pending : {Real(0),Real(3),Real(-2)}) {
            for (Real factor : {Real(1),Real(4),Real(0.5)}) {
                auto before = history_field(boxes,Real(10));
                auto after = history_field(boxes,current);
                auto baseline = history_field(boxes,Real(10)*factor-pending);
                erf_precipitation::rebase_precipitation_snapshot(baseline,before,after,factor);
                EXPECT_EQ(baseline.min(0),current*factor-pending);
                EXPECT_EQ(baseline.max(0),current*factor-pending);
            }
        }
    }
}

TEST(PrecipitationHistory, UnchangedValuesLeaveBaselineAndGhostsExact)
{
    const BoxArray boxes(history_box());
    auto before = history_field(boxes,Real(100));
    auto after = history_field(boxes,Real(100));
    // The small baseline would change if evaluated as 100-(100-baseline).
    const Real value = std::numeric_limits<Real>::epsilon()/Real(4);
    auto baseline = history_field(boxes,value,1);
    erf_precipitation::rebase_precipitation_snapshot(baseline,before,after,Real(1));
    EXPECT_EQ(baseline.min(0,1),value);
    EXPECT_EQ(baseline.max(0,1),value);
}

TEST(PrecipitationHistory, AlignsDecompositionsAndPreservesUncoveredCells)
{
    BoxArray old_boxes(history_box()); old_boxes.maxSize(IntVect(2,3,2));
    BoxArray new_boxes(history_box()); new_boxes.maxSize(IntVect(4,2,1));
    BoxArray baseline_boxes(history_box()); baseline_boxes.maxSize(IntVect(8,3,4));
    auto before = history_field(old_boxes,Real(10));
    auto after = history_field(new_boxes,Real(10));
    auto baseline = history_field(baseline_boxes,Real(7),1);
    for (MFIter mfi(after); mfi.isValid(); ++mfi) {
        const auto a=after.array(mfi);
        ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            if (i<4 && k==0) a(i,j,k)=Real(2);
        });
    }
    Gpu::streamSynchronizeAll();
    erf_precipitation::rebase_precipitation_snapshot(baseline,before,after,Real(1));
    MultiFab error(baseline.boxArray(),baseline.DistributionMap(),1,1);
    for (MFIter mfi(baseline); mfi.isValid(); ++mfi) {
        const auto a=baseline.const_array(mfi);
        const auto e=error.array(mfi);
        const Box valid=mfi.validbox();
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            const Real expected=valid.contains(IntVect(i,j,k)) && i<4 && k==0 ? Real(-1) : Real(7);
            e(i,j,k)=a(i,j,k)-expected;
        });
    }
    EXPECT_FALSE(error.contains_nan(0,1,1));
    EXPECT_EQ(error.norm0(0,1),Real(0));
}

TEST(PrecipitationHistory, CapturedSlmSourceOwnsOldSurfaceValuesAndConversion)
{
    const BoxArray boxes(history_box());
    auto rain=history_field(boxes,Real(7));
    auto snow=history_field(boxes,Real(11));
    SurfacePrecipAccumulationSources sources;
    sources.rain={&rain,Real(4)};
    sources.snow={&snow,Real(0.917)};
    const Geometry geometry(history_box(),RealBox(0,0,0,8,6,4),0,{0,0,0});
    auto saved=erf_precipitation::capture_slm_precipitation(sources,geometry);
    rain.setVal(Real(8)); snow.setVal(Real(12));
    ASSERT_NE(saved.values,nullptr);
    EXPECT_EQ(saved.values->min(0),Real(7));
    EXPECT_EQ(saved.values->max(0),Real(7));
    EXPECT_EQ(saved.values->boxArray()[0].length(2),1);
    EXPECT_EQ(saved.factor,Real(4));
}

TEST(PrecipitationHistory, SlmSelectsTotalThenRainAndIgnoresFrozenOnlySources)
{
    const Box domain(IntVect(-4,-3,3),IntVect(3,2,6));
    const Geometry geometry(domain,RealBox(0,0,0,8,6,4),0,{0,0,0});
    BoxArray atmospheric(domain); atmospheric.maxSize(IntVect(4,3,2));
    Box bottom=domain; bottom.setRange(2,0);
    BoxArray surface(bottom); surface.maxSize(IntVect(2,3,1));
    auto old_rain=history_field(atmospheric,Real(10));
    auto new_rain=history_field(atmospheric,Real(14));
    auto old_total=history_field(atmospheric,Real(20));
    auto new_total=history_field(atmospheric,Real(26));
    auto frozen=history_field(atmospheric,Real(999));
    for (int selection=0; selection<3; ++selection) {
        SurfacePrecipAccumulationSources before,after;
        before.snow={&frozen,Real(1000)}; after.snow=before.snow;
        if (selection<2) {
            before.rain={&old_rain,Real(2)}; after.rain={&new_rain,Real(2)};
        }
        if (selection==0) {
            before.total={&old_total,Real(4)}; after.total={&new_total,Real(4)};
        }
        auto baseline=history_field(surface,Real(7));
        auto saved=erf_precipitation::capture_slm_precipitation(before,geometry);
        erf_precipitation::rebase_slm_precipitation(baseline,saved,after,geometry);
        const Real expected=selection==0 ? Real(31) : selection==1 ? Real(15) : Real(7);
        EXPECT_EQ(baseline.min(0),expected);
        EXPECT_EQ(baseline.max(0),expected);
    }
}

TEST(PrecipitationHistory, BoundedRoundoffUsesCumulativeScale)
{
    const BoxArray boxes(history_box());
    for (Real scale : {Real(0.001),Real(1),Real(1000)}) {
        for (Real factor : {Real(0.917),Real(1),Real(1000)}) {
            for (int n=1;n<=20;++n) {
                const Real old=scale*Real(n)/Real(7);
                const Real current=scale*Real(21-n)/Real(3);
                const Real consumed=old*factor*Real(0.9);
                auto before=history_field(boxes,old);
                auto after=history_field(boxes,current);
                auto baseline=history_field(boxes,consumed);
                erf_precipitation::rebase_precipitation_snapshot(baseline,before,after,factor);
                const long double exact=static_cast<long double>(consumed)+
                    static_cast<long double>(factor)*(static_cast<long double>(current)-old);
                // Two products and two subtractions; bound roundoff by the
                // input cumulative magnitudes, even when their delta is tiny.
                const long double bound=8*std::numeric_limits<Real>::epsilon()*
                    (std::abs(static_cast<long double>(consumed))+
                     std::abs(static_cast<long double>(factor)*old)+
                     std::abs(static_cast<long double>(factor)*current));
                EXPECT_FALSE(baseline.contains_nan());
                EXPECT_LE(std::abs(static_cast<long double>(baseline.min(0))-exact),bound);
                EXPECT_LE(std::abs(static_cast<long double>(baseline.max(0))-exact),bound);
            }
        }
    }
}
