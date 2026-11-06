#ifndef ERF_GROUND_MASS_REMAP_H_
#define ERF_GROUND_MASS_REMAP_H_

#include "ERF_GroundMass.H"
#include "ERF_SurfacePrecipitationRemap.H"

namespace erf_ground_mass {

struct CapturedField
{
    std::string name;
    std::unique_ptr<amrex::MultiFab> surface;
};
using CapturedFields = amrex::Vector<CapturedField>;

inline CapturedFields capture (const GroundMassAccumulationFields& fields,
                               const amrex::Geometry& geometry)
{
    CapturedFields saved;
    for (const auto& field : fields) {
        AMREX_ALWAYS_ASSERT(field.mass != nullptr);
        saved.push_back({field.name,
            erf_precipitation::capture_surface_accumulation(*field.mass, geometry)});
    }
    return saved;
}

// Apportion a parent's kg equally over its horizontal children, then retain
// old valid fine values. New level creation passes no retained fields, even
// when the model still owns storage from an earlier incarnation of that level.
inline void restore (
    const GroundMassAccumulationFields& fields, const CapturedFields& retained,
    const amrex::Geometry& geometry,
    const GroundMassAccumulationFields& coarse_fields = {},
    const amrex::Geometry* coarse_geometry = nullptr,
    const amrex::IntVect& refinement = amrex::IntVect(1))
{
    using namespace amrex;
    AMREX_ALWAYS_ASSERT(retained.empty() || retained.size() == fields.size());
    AMREX_ALWAYS_ASSERT(!coarse_geometry || coarse_fields.size() == fields.size());
    const IntVect ratio(refinement[0], refinement[1], 1);
    AMREX_ALWAYS_ASSERT(ratio.allGT(IntVect(0)));
    if (fields.empty()) return;
#ifdef AMREX_USE_GPU
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!Gpu::inNoSyncRegion(),
        "Ground mass remapping requires GPU synchronization");
#endif
    for (int n = 0; n < fields.size(); ++n) {
        AMREX_ALWAYS_ASSERT(fields[n].mass != nullptr);
        if (!retained.empty()) AMREX_ALWAYS_ASSERT(retained[n].name == fields[n].name);
        auto& destination = *fields[n].mass;
        auto slice = erf_precipitation::make_surface_slice(destination, geometry);
        if (!slice.values) continue;
        slice.values->setVal(Real(0));
        Gpu::streamSynchronizeAll();
        if (coarse_geometry) {
            AMREX_ALWAYS_ASSERT(coarse_fields[n].name == fields[n].name);
            AMREX_ALWAYS_ASSERT(coarse_fields[n].mass != nullptr);
            auto source = erf_precipitation::capture_surface_accumulation(
                *coarse_fields[n].mass, *coarse_geometry);
            AMREX_ALWAYS_ASSERT(source != nullptr);
            AMREX_ALWAYS_ASSERT(slice.values->boxArray().coarsenable(ratio));
            BoxArray parent_boxes = slice.values->boxArray();
            parent_boxes.coarsen(ratio);
            MultiFab parents(parent_boxes, slice.values->DistributionMap(), 1, 0);
            parents.setVal(Real(0));
            Gpu::streamSynchronizeAll();
            parents.ParallelCopy(*source, 0, 0, 1, 0, 0,
                erf_precipitation::surface_geometry(*coarse_geometry).periodicity());
            Gpu::streamSynchronizeAll();
            const int rx = refinement[0], ry = refinement[1];
            const Real children = Real(rx)*Real(ry);
            for (MFIter mfi(*slice.values, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
                const auto from = parents.const_array(mfi);
                const auto to = slice.values->array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int) noexcept {
                    to(i,j,0) = from(amrex::coarsen(i,rx),amrex::coarsen(j,ry),0)/children;
                });
            }
        }
        if (!retained.empty() && retained[n].surface) {
            slice.values->ParallelCopy(*retained[n].surface, 0, 0, 1, 0, 0,
                erf_precipitation::surface_geometry(geometry).periodicity());
            Gpu::streamSynchronizeAll();
        }
        const int bottom = geometry.Domain().smallEnd(2);
        for (MFIter mfi(*slice.values, MFItInfo().SetDeviceSync(true)); mfi.isValid(); ++mfi) {
            const auto from = slice.values->const_array(mfi);
            const auto to = destination.array(slice.atmospheric_indices[mfi.index()]);
            ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int) noexcept {
                to(i,j,bottom) = from(i,j,0);
            });
        }
    }
}

} // namespace erf_ground_mass
#endif
