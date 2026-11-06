#ifndef ERF_PRECIPITATION_HISTORY_H_
#define ERF_PRECIPITATION_HISTORY_H_

#include "ERF_SurfacePrecipitationRemap.H"

namespace erf_precipitation {

// SLM consumes the total source when available, otherwise liquid rain. Retain
// only that physical surface plane while restriction changes its borrowed view.
struct CapturedSlmPrecipitation
{
    std::unique_ptr<amrex::MultiFab> values;
    amrex::Real factor = 0;
};

inline CapturedSlmPrecipitation
capture_slm_precipitation (const SurfacePrecipAccumulationSources& sources,
                          const amrex::Geometry& geometry)
{
    const auto& selected = surface_precip_has_total_source(sources) ? sources.total : sources.rain;
    CapturedSlmPrecipitation saved;
    if (selected.accum) {
        AMREX_ALWAYS_ASSERT(selected.accum->nComp() == 1);
        saved.values = capture_surface_accumulation(*selected.accum, geometry);
        saved.factor = selected.native_to_kg_m2;
    }
    return saved;
}

// Change the cumulative reference while retaining its unconsumed interval.
// Native-unit snapshots use factor=1; SLM uses its typed kg/m2 conversion.
inline void rebase_precipitation_snapshot (
    amrex::MultiFab& baseline, const amrex::MultiFab& before,
    const amrex::MultiFab& after, amrex::Real factor)
{
    using namespace amrex;
    AMREX_ALWAYS_ASSERT(baseline.nComp() == 1 && before.nComp() == 1 && after.nComp() == 1);
    MultiFab old_values(baseline.boxArray(), baseline.DistributionMap(), 1, 0);
    MultiFab new_values(baseline.boxArray(), baseline.DistributionMap(), 1, 0);
    old_values.setVal(Real(0));
    new_values.setVal(Real(0));
    old_values.ParallelCopy(before, 0, 0, 1, 0, 0);
    new_values.ParallelCopy(after, 0, 0, 1, 0, 0);
    Gpu::streamSynchronizeAll();
    for (MFIter mfi(baseline, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
        const auto prior = old_values.const_array(mfi);
        const auto current = new_values.const_array(mfi);
        const auto consumed = baseline.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
            // Leave uncovered/unchanged cells bitwise unchanged. Evaluating
            // old-(old-baseline) could otherwise round a valid baseline.
            if (prior(i,j,k) != current(i,j,k)) {
                const Real pending = prior(i,j,k)*factor - consumed(i,j,k);
                consumed(i,j,k) = current(i,j,k)*factor - pending;
            }
        });
    }
}

inline void rebase_slm_precipitation (
    amrex::MultiFab& baseline, const CapturedSlmPrecipitation& before,
    const SurfacePrecipAccumulationSources& after, const amrex::Geometry& geometry)
{
    const auto& new_source = surface_precip_has_total_source(after) ? after.total : after.rain;
    if (!new_source.accum) {
        AMREX_ALWAYS_ASSERT(!before.values);
        return;
    }
    auto new_surface = capture_surface_accumulation(*new_source.accum, geometry);
    AMREX_ALWAYS_ASSERT(bool(before.values) == bool(new_surface));
    if (!new_surface) return;
    AMREX_ALWAYS_ASSERT(before.factor == new_source.native_to_kg_m2);
    rebase_precipitation_snapshot(baseline, *before.values, *new_surface, before.factor);
}

} // namespace erf_precipitation
#endif
