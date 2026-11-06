#ifndef ERF_SURFACE_PRECIPITATION_REMAP_H_
#define ERF_SURFACE_PRECIPITATION_REMAP_H_

#include <AMReX_BCUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFabUtil.H>

#include "ERF_SurfacePrecipitation.H"

namespace erf_precipitation {

// Native accumulators are stored on the atmospheric 3D layout, but only the
// physical bottom plane carries surface precipitation. Keep one disjoint copy
// of each surface column, even when the atmosphere is decomposed vertically.
struct SurfaceSlice
{
    std::unique_ptr<amrex::MultiFab> values;
    amrex::Vector<int> atmospheric_indices;
};

inline SurfaceSlice
make_surface_slice (const amrex::MultiFab& atmospheric,
                    const amrex::Geometry& geometry,
                    const amrex::IntVect& ghosts = amrex::IntVect(0))
{
    using namespace amrex;
    SurfaceSlice slice;
    BoxList boxes(IndexType::TheCellType());
    Vector<int> processors;
    const int bottom = geometry.Domain().smallEnd(2);
    for (int ib = 0; ib < atmospheric.boxArray().size(); ++ib) {
        Box box = atmospheric.boxArray()[ib];
        if (box.smallEnd(2) <= bottom && box.bigEnd(2) >= bottom) {
            box.setRange(2, 0);
            boxes.push_back(box);
            processors.push_back(atmospheric.DistributionMap()[ib]);
            slice.atmospheric_indices.push_back(ib);
        }
    }
    if (!boxes.isEmpty()) {
        BoxArray ba(std::move(boxes));
        AMREX_ALWAYS_ASSERT(ba.isDisjoint());
        slice.values = std::make_unique<MultiFab>(ba,
            DistributionMapping(std::move(processors)), 1, ghosts);
    }
    return slice;
}

inline amrex::Geometry
surface_geometry (const amrex::Geometry& geometry)
{
    amrex::Box domain = geometry.Domain();
    domain.setRange(2, 0);
    return amrex::Geometry(domain, geometry.ProbDomain(), geometry.Coord(),
                           {geometry.isPeriodic(0), geometry.isPeriodic(1), 0});
}

inline std::unique_ptr<amrex::MultiFab>
capture_surface_accumulation (const amrex::MultiFab& source,
                              const amrex::Geometry& geometry,
                              const amrex::IntVect& ghosts = amrex::IntVect(0))
{
    using namespace amrex;
    auto slice = make_surface_slice(source, geometry, ghosts);
    if (!slice.values) { return nullptr; }
    const int bottom = geometry.Domain().smallEnd(2);
    for (MFIter mfi(*slice.values, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
        const auto from = source.const_array(slice.atmospheric_indices[mfi.index()]);
        const auto to = slice.values->array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {
            to(i,j,0) = from(i,j,bottom);
        });
    }
    // The iterator completes every used stream before Init can destroy source.
    return std::move(slice.values);
}

// Restrict a surface depth by its horizontal area, without averaging in empty
// atmospheric cells above it. Leave uncovered coarse columns and all upper
// cells unchanged. Fine patches wholly above the ground have nothing to add.
inline void
average_down_surface_accumulation (
    const amrex::MultiFab& fine, amrex::MultiFab& coarse,
    const amrex::Geometry& fine_geometry, const amrex::Geometry& coarse_geometry,
    const amrex::IntVect& refinement)
{
    using namespace amrex;
#ifdef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!Gpu::inNoSyncRegion(),
        "Surface precipitation restriction requires GPU synchronization");
#endif
    auto fine_surface = capture_surface_accumulation(fine, fine_geometry);
    if (!fine_surface) { return; }
    auto coarse_surface = capture_surface_accumulation(coarse, coarse_geometry);
    AMREX_ALWAYS_ASSERT(coarse_surface != nullptr);
    const IntVect surface_ratio(refinement[0], refinement[1], 1);
    average_down(*fine_surface, *coarse_surface, 0, 1, surface_ratio);
    Gpu::streamSynchronizeAll();

    const auto layout = make_surface_slice(coarse, coarse_geometry);
    const int bottom = coarse_geometry.Domain().smallEnd(2);
    for (MFIter mfi(*coarse_surface, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
        const auto from = coarse_surface->const_array(mfi);
        const auto to = coarse.array(layout.atmospheric_indices[mfi.index()]);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {
            to(i,j,bottom) = from(i,j,0);
        });
    }
}

// Species and aerosol ground accumulators store mass per horizontal cell.
// Sum their children rather than averaging depths or atmospheric volumes.
inline void
sum_down_surface_mass (
    const amrex::MultiFab& fine, amrex::MultiFab& coarse,
    const amrex::Geometry& fine_geometry, const amrex::Geometry& coarse_geometry,
    const amrex::IntVect& refinement)
{
    using namespace amrex;
#ifdef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!Gpu::inNoSyncRegion(),
        "Surface mass restriction requires GPU synchronization");
#endif
    auto fine_surface = capture_surface_accumulation(fine, fine_geometry);
    if (!fine_surface) { return; }
    const IntVect surface_ratio(refinement[0], refinement[1], 1);
    AMREX_ALWAYS_ASSERT(surface_ratio.allGT(IntVect(0)));
    AMREX_ALWAYS_ASSERT(fine_surface->boxArray().coarsenable(surface_ratio));
    BoxArray restricted_boxes = fine_surface->boxArray();
    restricted_boxes.coarsen(surface_ratio);
    MultiFab restricted(restricted_boxes, fine_surface->DistributionMap(), 1, 0);
    const int rx = refinement[0];
    const int ry = refinement[1];
    for (MFIter mfi(restricted, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
        const auto from = fine_surface->const_array(mfi);
        const auto to = restricted.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {
            Real mass = 0;
            for (int jj = 0; jj < ry; ++jj) {
                for (int ii = 0; ii < rx; ++ii) {
                    mass += from(i*rx+ii,j*ry+jj,0);
                }
            }
            to(i,j,0) = mass;
        });
    }
    auto coarse_surface = capture_surface_accumulation(coarse, coarse_geometry);
    AMREX_ALWAYS_ASSERT(coarse_surface != nullptr);
    coarse_surface->ParallelCopy(restricted, 0, 0, 1, 0, 0);
    Gpu::streamSynchronizeAll();

    const auto layout = make_surface_slice(coarse, coarse_geometry);
    const int bottom = coarse_geometry.Domain().smallEnd(2);
    for (MFIter mfi(*coarse_surface, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
        const auto from = coarse_surface->const_array(mfi);
        const auto to = coarse.array(layout.atmospheric_indices[mfi.index()]);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {
            to(i,j,bottom) = from(i,j,0);
        });
    }
}

inline bool
is_native_surface_accumulator (const amrex::MultiFab* field,
                               const SurfacePrecipAccumulationSources& sources)
{
    return field && (field == sources.total.accum || field == sources.rain.accum ||
                     field == sources.snow.accum || field == sources.graupel.accum ||
                     field == sources.hail.accum);
}

// Call after the scheme has initialized its new native arrays. New fine cells
// inherit the coarse cumulative history; old valid fine cells retain their own.
// No atmospheric state or source halo is copied, and refinement in z must not
// dilute an accumulation that represents a surface rather than a volume.
inline void
restore_native_surface_accumulation (
    amrex::MultiFab& destination,
    const amrex::MultiFab* retained_surface,
    const amrex::Geometry& geometry,
    const amrex::MultiFab* coarse_native = nullptr,
    const amrex::Geometry* coarse_geometry = nullptr,
    const amrex::IntVect& refinement = amrex::IntVect(1))
{
    using namespace amrex;
#ifdef AMREX_USE_GPU
    // AMReX's interpolator owns temporaries whose MFIter completion depends on
    // the ordinary synchronized region used by ERF's level lifecycle.
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!coarse_native || !Gpu::inNoSyncRegion(),
        "Surface precipitation interpolation requires GPU synchronization");
#endif
    auto slice = make_surface_slice(destination, geometry);
    if (!slice.values) { return; }
    slice.values->setVal(Real(0));
    Gpu::streamSynchronizeAll();
    const auto fine_surface_geom = surface_geometry(geometry);
    if (coarse_native) {
        AMREX_ALWAYS_ASSERT(coarse_geometry != nullptr);
        const auto coarse_surface_geom = surface_geometry(*coarse_geometry);
        auto coarse = capture_surface_accumulation(*coarse_native, *coarse_geometry,
                                                   IntVect(1,1,0));
        AMREX_ALWAYS_ASSERT(coarse != nullptr);
        // Accumulation halos are not evolved by microphysics. Rebuild them
        // from current valid cells before the interpolation reads its slopes.
        Vector<BCRec> bcs(1);
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int bc = coarse_surface_geom.isPeriodic(dir)
                ? BCType::int_dir : BCType::foextrap;
            bcs[0].setLo(dir, bc);
            bcs[0].setHi(dir, bc);
        }
        coarse->FillBoundary(coarse_surface_geom.periodicity());
        Gpu::streamSynchronizeAll();
        FillDomainBoundary(*coarse, coarse_surface_geom, bcs);
        Gpu::streamSynchronizeAll();
        const IntVect surface_ratio(refinement[0], refinement[1], 1);
        InterpFromCoarseLevel(*slice.values, IntVect(0), IntVect(0),
                             *coarse, 0, 0, 1,
                             coarse_surface_geom, fine_surface_geom,
                             surface_ratio, &cell_cons_interp, bcs, 0);
        Gpu::streamSynchronizeAll(); // interpolation must finish before coarse is destroyed
    }
    if (retained_surface) {
        slice.values->ParallelCopy(*retained_surface, 0, 0, 1, 0, 0,
                                    fine_surface_geom.periodicity());
        Gpu::streamSynchronizeAll();
    }
    const int bottom = geometry.Domain().smallEnd(2);
    for (MFIter mfi(*slice.values, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
        const auto from = slice.values->const_array(mfi);
        const auto to = destination.array(slice.atmospheric_indices[mfi.index()]);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {
            to(i,j,bottom) = from(i,j,0);
        });
    }
    // The iterator completes the scatter before temporary surface storage dies.
}

// Seed newly covered SLM cells from the same remapped native field they will
// difference on the next step. The caller then overlays the old valid SLM
// baseline, preserving its interval history. Match AdvanceLSM's total-or-rain
// selection and unit conversion; explicit frozen species do not feed SLM.
inline void
seed_slm_baseline_from_surface (amrex::MultiFab& baseline,
                                const SurfacePrecipAccumulationSources& sources,
                                const amrex::Geometry& geometry)
{
    const auto& source = surface_precip_has_total_source(sources)
        ? sources.total : sources.rain;
    if (!source.accum) { return; }
    auto surface = capture_surface_accumulation(*source.accum, geometry);
    if (!surface) { return; }
    surface->mult(source.native_to_kg_m2, 0, 1, 0);
    amrex::Gpu::streamSynchronizeAll();
    baseline.ParallelCopy(*surface, 0, 0, 1, 0, 0,
                           surface_geometry(geometry).periodicity());
    amrex::Gpu::streamSynchronizeAll(); // complete the copy before releasing surface
}

} // namespace erf_precipitation

#endif
