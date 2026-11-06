#include <array>
#include <filesystem>
#include <memory>
#include <string>

#include <AMReX_MultiFab.H>
#include <AMReX_Utility.H>
#include <AMReX_VisMF.H>
#include <gtest/gtest.h>

#include "ERF_NOAHMP.H"
#include "../../ERF_GTestTempDir.H"

namespace {

using amrex::MultiFab;
using amrex::Real;

// These tests exercise the real C++ bookkeeping and VisMF I/O with an empty
// Fortran block vector. They do not initialize or execute Noah-MP physics.
class NoahMPPrecipSnapshots : public ::testing::Test {
protected:
    std::filesystem::path root;

    void SetUp () override {
        root = erf_gtest_temp_path("erf_noahmp_precip_snapshots");
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::create_directories(root);
        }
        amrex::ParallelDescriptor::Barrier();
    }

    void TearDown () override {
        amrex::ParallelDescriptor::Barrier();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::remove_all(root);
        }
    }

    std::string directory (const std::string& name) {
        auto path = root / name;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::filesystem::create_directories(path);
        }
        amrex::ParallelDescriptor::Barrier();
        return path.string();
    }

    static std::unique_ptr<MultiFab> field (Real value, bool split = false) {
        amrex::BoxArray boxes(amrex::Box(amrex::IntVect(0), amrex::IntVect(3,3,0)));
        if (split) { boxes.maxSize(2); }
        auto result = std::make_unique<MultiFab>(boxes, amrex::DistributionMapping(boxes), 1, 0);
        result->setVal(value);
        return result;
    }

    static SurfacePrecipAccumulationSources rain_source (const MultiFab& rain) {
        SurfacePrecipAccumulationSources sources;
        sources.rain = {&rain, Real(1)};
        return sources;
    }

    static std::string prefix (const std::string& dir, int slot) {
        return NOAHMP::m_precip_snapshot_name(dir, 0, slot);
    }

    static bool present (const std::string& dir, int slot) {
        return amrex::FileExists(prefix(dir, slot) + "_H");
    }

    static void expect_saved (const std::string& dir, int slot, Real expected) {
        ASSERT_TRUE(present(dir,slot));
        MultiFab saved;
        amrex::VisMF::Read(saved, prefix(dir,slot));
        // Uniform exact integers are serialized without conversion or arithmetic.
        EXPECT_EQ(saved.min(0), expected);
        EXPECT_EQ(saved.max(0), expected);
    }
};

// No atmospheric advance occurs between Read and Write. Every saved category
// must survive repeated restart-only checkpoint output.
TEST_F(NoahMPPrecipSnapshots, ImmediateCheckpointPreservesEveryRestoredSlot)
{
    std::array<std::unique_ptr<MultiFab>, NoahmpPrecipSlot::NumSlots> fields;
    for (int s=0; s<NoahmpPrecipSlot::NumSlots; ++s) { fields[s] = field(8+s); }
    SurfacePrecipAccumulationSources sources;
    sources.total = {fields[NoahmpPrecipSlot::total].get(), Real(1)};
    sources.rain = {fields[NoahmpPrecipSlot::rain].get(), Real(1)};
    sources.snow = {fields[NoahmpPrecipSlot::snow].get(), Real(1)};
    sources.graupel = {fields[NoahmpPrecipSlot::graupel].get(), Real(1)};
    sources.hail = {fields[NoahmpPrecipSlot::hail].get(), Real(1)};
    NOAHMP initial;
    initial.prepare_precip_snapshots(0, initial.collect_precip_sources(sources));
    auto previous = directory("initial");
    initial.Write_Lsm_Restart(previous);
    for (int round=0; round<2; ++round) {
        NOAHMP restored;
        restored.Read_Lsm_Restart(previous);
        const auto next = directory("round" + std::to_string(round));
        restored.Write_Lsm_Restart(next);
        for (int s=0; s<NoahmpPrecipSlot::NumSlots; ++s) { expect_saved(next,s,8+s); }
        previous = next;
    }
}

// Restored precipitation predates the current atmospheric step. Preparation
// must retain that unconsumed interval and remap a changed box decomposition.
TEST_F(NoahMPPrecipSnapshots, RestoredBaselineSurvivesPreparationUntilConsumption)
{
    auto rain = field(9);
    NOAHMP initial;
    initial.prepare_precip_snapshots(0, initial.collect_precip_sources(rain_source(*rain)));
    const auto first = directory("first");
    initial.Write_Lsm_Restart(first);

    NOAHMP restored;
    restored.Read_Lsm_Restart(first);
    rain = field(12, true);
    auto slots = restored.collect_precip_sources(rain_source(*rain));
    restored.prepare_precip_snapshots(0,slots);
    rain->setVal(14);
    restored.prepare_precip_snapshots(0,slots);
    const auto unconsumed = directory("unconsumed");
    restored.Write_Lsm_Restart(unconsumed);
    expect_saved(unconsumed,NoahmpPrecipSlot::rain,9);
    MultiFab baseline;
    amrex::VisMF::Read(baseline,prefix(unconsumed,NoahmpPrecipSlot::rain));
    EXPECT_EQ(rain->min(0)-baseline.min(0), Real(5));

    restored.advance_precip_snapshots(0,slots);
    const auto consumed = directory("consumed");
    restored.Write_Lsm_Restart(consumed);
    expect_saved(consumed,NoahmpPrecipSlot::rain,14);
}

// The pre-step hook invokes this public helper before microphysics changes the
// borrowed fields. Repeated preparation on skipped land firings must not retire
// the baseline, and inherited nonzero cumulative history must not be replayed.
TEST_F(NoahMPPrecipSnapshots, PrestepSeedIsIdempotentForTypedNativeSources)
{
    for (const int inherited_history : {0,12}) {
        auto rain = field(inherited_history);
        auto snow = field(4);
        SurfacePrecipAccumulationSources sources = rain_source(*rain);
        sources.snow = {snow.get(), Real(0.5)};
        NOAHMP model;
        auto slots = model.collect_precip_sources(sources);
        EXPECT_EQ(slots.factor[NoahmpPrecipSlot::snow], Real(0.5));
        model.prepare_precip_snapshots(0,slots);
        rain->setVal(inherited_history+2);
        snow->setVal(5);
        model.prepare_precip_snapshots(0,slots);
        const auto skipped = directory("skipped"+std::to_string(inherited_history));
        model.Write_Lsm_Restart(skipped);
        expect_saved(skipped,NoahmpPrecipSlot::rain,inherited_history);
        expect_saved(skipped,NoahmpPrecipSlot::snow,4);
        MultiFab rain_prev, snow_prev;
        amrex::VisMF::Read(rain_prev,prefix(skipped,NoahmpPrecipSlot::rain));
        amrex::VisMF::Read(snow_prev,prefix(skipped,NoahmpPrecipSlot::snow));
        EXPECT_EQ((rain->min(0)-rain_prev.min(0))*slots.factor[NoahmpPrecipSlot::rain], Real(2));
        EXPECT_EQ((snow->min(0)-snow_prev.min(0))*slots.factor[NoahmpPrecipSlot::snow], Real(0.5));
        EXPECT_FALSE(present(skipped,NoahmpPrecipSlot::total));
        EXPECT_FALSE(present(skipped,NoahmpPrecipSlot::graupel));
        EXPECT_FALSE(present(skipped,NoahmpPrecipSlot::hail));
    }
}

TEST_F(NoahMPPrecipSnapshots, AbsentSourcesProduceNoSnapshots)
{
    NOAHMP model;
    model.prepare_precip_snapshots(0,model.collect_precip_sources({}));
    const auto output = directory("dry");
    model.Write_Lsm_Restart(output);
    for (int s=0; s<NoahmpPrecipSlot::NumSlots; ++s) { EXPECT_FALSE(present(output,s)); }
}

} // namespace
