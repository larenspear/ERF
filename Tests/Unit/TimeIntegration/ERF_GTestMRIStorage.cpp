#include <memory>
#include <utility>

#include <AMReX_MFIter.H>
#include <gtest/gtest.h>

#include "ERF_MRI.H"

// MRI storage must outlive its sizing template and follow the current layout
// after initialize().  Exercise the real AMReX allocation and callback paths.
namespace {

using State = amrex::Vector<amrex::MultiFab>;
using Integrator = MRISplitIntegrator<State>;

State make_state (int size, int components, amrex::IntVect ghosts)
{
    using namespace amrex;
    BoxArray boxes(Box(IntVect(0), IntVect(size-1)));
    boxes.maxSize(4);
    DistributionMapping mapping(boxes);
    State state;
    state.emplace_back(boxes, mapping, components, ghosts);
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        IntVect nodal(0);
        nodal[dir] = 1;
        state.emplace_back(convert(boxes, nodal), mapping, 1, ghosts);
    }
    for (auto& mf : state) { mf.setVal(amrex::Real(7)); }
    return state;
}

void expect_layout (const State& actual, const State& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (int n = 0; n < expected.size(); ++n) {
        EXPECT_EQ(actual[n].boxArray(), expected[n].boxArray());
        EXPECT_EQ(actual[n].DistributionMap(), expected[n].DistributionMap());
        EXPECT_EQ(actual[n].nComp(), expected[n].nComp());
        EXPECT_EQ(actual[n].nGrowVect(), expected[n].nGrowVect());
    }
}

void expect_zero (const State& state)
{
    for (auto const& mf : state) {
        // Include every component and every allocated ghost cell.
        EXPECT_EQ(mf.norm0(0, mf.nComp(), mf.nGrowVect()), amrex::Real(0));
    }
}

// advance() exposes both active scratch states through the public callbacks.
// Match them to map_data()'s two owned states, without adding test-only accessors.
void expect_active_storage (Integrator& integrator, State& state)
{
    amrex::Vector<State*> stored;
    integrator.map_data([&] (State& data) { stored.push_back(&data); });
    ASSERT_EQ(stored.size(), 2u);

    // Build an exact layout match, including tests that change box/component counts.
    State next;
    for (auto const& mf : state) {
        next.emplace_back(mf.boxArray(), mf.DistributionMap(), mf.nComp(), mf.nGrowVect());
    }

    integrator.setNcompCons(state[0].nComp());
    integrator.setAnelastic(0);
    integrator.setNoSubstepping(1);
    integrator.setForceFirstStageSingleSubstep(1);

    int pre_calls = 0;
    int no_substep_calls = 0;
    int post_calls = 0;
    integrator.set_slow_rhs_pre([&] (State& rhs, State&, State&, double, double, double, int) {
        EXPECT_EQ(&rhs, stored[1]);
        expect_layout(rhs, state);
        ++pre_calls;
    });
    integrator.set_no_substep([&] (State& sum, State&, State& rhs, double, double, int) {
        EXPECT_EQ(&sum, stored[0]);
        EXPECT_EQ(&rhs, stored[1]);
        expect_layout(sum, state);
        ++no_substep_calls;
    });
    integrator.set_slow_rhs_post([&] (State& rhs, State&, State&, State& sum,
                                     double, double, double, int) {
        EXPECT_EQ(&sum, stored[0]);
        EXPECT_EQ(&rhs, stored[1]);
        ++post_calls;
    });
    EXPECT_EQ(integrator.advance(state, next, 0.0, 1.0), 1.0);
    EXPECT_EQ(pre_calls, 3);
    EXPECT_EQ(no_substep_calls, 3);
    EXPECT_EQ(post_calls, 3);

    // The callbacks borrow this function's locals; release them before returning.
    integrator.set_slow_rhs_pre({});
    integrator.set_no_substep({});
    integrator.set_slow_rhs_post({});
}

} // namespace

TEST(MRIStorage, ConstructorAllocatesIndependentScratchWithGhosts)
{
    auto state = make_state(8, 3, amrex::IntVect(2, 1, 0));
    Integrator integrator(state);
    amrex::Vector<State*> stored;
    integrator.map_data([&] (State& data) {
        expect_layout(data, state);
        expect_zero(data);
        stored.push_back(&data);
    });
    ASSERT_EQ(stored.size(), 2u);
    for (int n = 0; n < state.size(); ++n) {
        for (amrex::MFIter mfi(state[n]); mfi.isValid(); ++mfi) {
            EXPECT_NE((*stored[0])[n][mfi].dataPtr(), state[n][mfi].dataPtr());
            EXPECT_NE((*stored[1])[n][mfi].dataPtr(), state[n][mfi].dataPtr());
            EXPECT_NE((*stored[0])[n][mfi].dataPtr(), (*stored[1])[n][mfi].dataPtr());
        }
        (*stored[0])[n].setVal(amrex::Real(11));
        EXPECT_EQ(state[n].min(0), amrex::Real(7));
        EXPECT_EQ(state[n].max(0), amrex::Real(7));
    }
    expect_zero(*stored[1]);
    expect_active_storage(integrator, state);
}

TEST(MRIStorage, InitializeReplacesBothScratchLayouts)
{
    auto original = make_state(4, 2, amrex::IntVect(1));
    Integrator integrator(original);
    for (int size : {8, 4, 12}) {
        auto replacement = make_state(size, size/2, amrex::IntVect(2, 0, 1));
        integrator.initialize(replacement);
        amrex::Vector<State*> stored;
        integrator.map_data([&] (State& data) { stored.push_back(&data); });
        // Diagnose the two active layouts before rejecting extra allocations;
        // those extra allocations were uninitialized in the broken version.
        ASSERT_GE(stored.size(), 2u);
        expect_layout(*stored[0], replacement);
        expect_layout(*stored[1], replacement);
        ASSERT_EQ(stored.size(), 2u);
        for (auto* data : stored) {
            expect_zero(*data);
            for (auto& mf : *data) { mf.setVal(amrex::Real(19)); }
        }
        expect_active_storage(integrator, replacement);
    }
}

TEST(MRIStorage, ScratchOutlivesTemplateAndMoveSource)
{
    std::unique_ptr<Integrator> moved;
    {
        auto temporary = make_state(8, 3, amrex::IntVect(1, 0, 2));
        Integrator source(temporary);
        moved = std::make_unique<Integrator>(std::move(source));
    }
    auto state = make_state(8, 3, amrex::IntVect(1, 0, 2));
    moved->map_data([&] (State& data) {
        expect_layout(data, state);
        expect_zero(data);
        for (auto& mf : data) { mf.setVal(amrex::Real(23)); }
    });
    expect_active_storage(*moved, state);

    auto previous = make_state(4, 2, amrex::IntVect(0));
    Integrator assigned(previous);
    assigned = std::move(*moved);
    moved.reset();
    expect_active_storage(assigned, state);
}

TEST(MRIStorage, InitializeCanUseAnOwnedScratchAsItsTemplate)
{
    auto state = make_state(8, 3, amrex::IntVect(1, 0, 2));
    Integrator integrator;
    integrator.initialize(state);
    State* borrowed = nullptr;
    integrator.map_data([&] (State& data) {
        borrowed = &data;
        for (auto& mf : data) { mf.setVal(amrex::Real(29)); }
    });
    ASSERT_NE(borrowed, nullptr);
    // Do this after map_data returns; replacing storage inside its traversal
    // would invalidate that traversal's iterators.
    integrator.initialize(*borrowed);
    borrowed = nullptr;
    int count = 0;
    integrator.map_data([&] (State& data) {
        expect_layout(data, state);
        ++count;
    });
    ASSERT_EQ(count, 2);
    integrator.map_data([&] (State& data) { expect_zero(data); });
    expect_active_storage(integrator, state);
}
