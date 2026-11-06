#include <gtest/gtest.h>
#include <cfenv>
#include <limits>

#include "ERF_SurfacePrecipitationRemap.H"

namespace {
using namespace amrex;

Geometry mass_geometry (const Box& domain)
{
    return Geometry(domain, RealBox(0,0,0,8,6,4), 0, {0,0,0});
}

MultiFab mass_field (const BoxArray& boxes, Real initial)
{
    MultiFab result(boxes, DistributionMapping(boxes), 1, 1);
    result.setVal(initial);
    return result;
}

void set_mass (MultiFab& field, int bottom, Real value, bool varying = false)
{
    for (MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto a = field.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            if (k == bottom) a(i,j,k) = value + (varying ? Real(2*i+3*j) : Real(0));
        });
    }
    Gpu::streamSynchronize();
}

template<class Expected>
void expect_mass (const MultiFab& field, Expected expected)
{
    MultiFab error(field.boxArray(), field.DistributionMap(), 1, 1);
    for (MFIter mfi(field); mfi.isValid(); ++mfi) {
        const auto a = field.const_array(mfi);
        const auto e = error.array(mfi);
        const Box valid = mfi.validbox();
        ParallelFor(mfi.fabbox(), [=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            e(i,j,k) = a(i,j,k) - expected(i,j,k,valid.contains(IntVect(i,j,k)));
        });
    }
    EXPECT_FALSE(error.contains_nan(0,1,1));
    EXPECT_EQ(error.norm0(0,1), Real(0));
}
} // namespace

TEST(SurfaceMassRestriction, SumsChildrenAndPreservesUncoveredUpperAndGhostCells)
{
    for (int rz : {1,2,3}) {
        const IntVect rr(2,3,rz);
        const Box cd(IntVect(-4,-3,3),IntVect(3,2,6));
        const Box fd = refine(cd,rr);
        const auto cg = mass_geometry(cd);
        const auto fg = mass_geometry(fd);
        BoxArray cb(cd); cb.maxSize(IntVect(2,3,2));
        BoxArray fb(refine(Box(IntVect(-2,-3,3),IntVect(1,2,6)),rr));
        fb.maxSize(IntVect(4,9,2*rz));
        auto coarse = mass_field(cb,Real(-17));
        auto fine = mass_field(fb,Real(-999));
        set_mass(coarse,3,Real(7));
        set_mass(fine,3*rz,Real(100),true);
        for (int repeat=0; repeat<2; ++repeat) {
            erf_precipitation::sum_down_surface_mass(fine,coarse,fg,cg,rr);
            expect_mass(coarse, [=] AMREX_GPU_HOST_DEVICE(int i,int j,int k,bool valid) noexcept {
                if (!valid || k!=3) return Real(-17);
                if (i < -2 || i > 1) return Real(7);
                return Real(624+24*i+54*j);
            });
        }
        expect_mass(fine, [=] AMREX_GPU_HOST_DEVICE(int i,int j,int k,bool valid) noexcept {
            return valid && k==3*rz ? Real(100+2*i+3*j) : Real(-999);
        });
    }
}

TEST(SurfaceMassRestriction, ElevatedPatchDoesNotChangeGroundHistory)
{
    const IntVect rr(2,3,2);
    const Box cd(IntVect(0),IntVect(7,5,3));
    auto coarse = mass_field(BoxArray(cd),Real(7));
    auto fine = mass_field(BoxArray(refine(Box(IntVect(0,0,2),IntVect(7,5,3)),rr)),Real(99));
    erf_precipitation::sum_down_surface_mass(fine,coarse,mass_geometry(refine(cd,rr)),mass_geometry(cd),rr);
    EXPECT_EQ(coarse.min(0,1),Real(7));
    EXPECT_EQ(coarse.max(0,1),Real(7));
}

TEST(SurfaceMassRestriction, ComposesAcrossThreeLevels)
{
    const IntVect first(2,3,2), second(3,2,3);
    const Box d0(IntVect(0,0,3),IntVect(1,1,4));
    const Box d1=refine(d0,first), d2=refine(d1,second);
    auto f0=mass_field(BoxArray(d0),Real(-7));
    auto f1=mass_field(BoxArray(d1),Real(-8));
    auto f2=mass_field(BoxArray(d2),Real(-9));
    set_mass(f2,d2.smallEnd(2),Real(2));
    erf_precipitation::sum_down_surface_mass(f2,f1,mass_geometry(d2),mass_geometry(d1),second);
    erf_precipitation::sum_down_surface_mass(f1,f0,mass_geometry(d1),mass_geometry(d0),first);
    expect_mass(f0, [] AMREX_GPU_HOST_DEVICE(int,int,int k,bool valid) noexcept {
        return valid && k==3 ? Real(72) : Real(-7);
    });
}

TEST(SurfaceMassRestriction, HorizontalIdentityAndOneDirectionRefinement)
{
    for (const auto rr : {IntVect(1,1,3),IntVect(1,3,2),IntVect(3,1,1)}) {
        const Box cd(IntVect(0),IntVect(3,3,1)), fd=refine(cd,rr);
        auto coarse=mass_field(BoxArray(cd),Real(0));
        auto fine=mass_field(BoxArray(fd),Real(0));
        set_mass(fine,0,Real(2));
        erf_precipitation::sum_down_surface_mass(fine,coarse,mass_geometry(fd),mass_geometry(cd),rr);
        const Real expected=Real(2*rr[0]*rr[1]);
        expect_mass(coarse, [=] AMREX_GPU_HOST_DEVICE(int,int,int k,bool valid) noexcept {
            return valid && k==0 ? expected : Real(0);
        });
    }
}

TEST(SurfaceMassRestriction, DoesNotScaleFiniteMassBeforeSumming)
{
    const IntVect rr(2,2,2);
    const Box cd(IntVect(0),IntVect(1,0,1)), fd=refine(cd,rr);
    auto coarse=mass_field(BoxArray(cd),Real(0));
    auto fine=mass_field(BoxArray(fd),Real(0));
    const Real large=std::numeric_limits<Real>::max()/Real(2);
    const Real tiny=std::numeric_limits<Real>::denorm_min();
    for (MFIter mfi(fine);mfi.isValid();++mfi) {
        const auto a=fine.array(mfi);
        ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            if (j==0 && k==0) {
                if (i==0) a(i,j,k)=large;
                if (i==2) a(i,j,k)=tiny;
            }
        });
    }
    Gpu::streamSynchronize();
    std::feclearexcept(FE_ALL_EXCEPT);
    erf_precipitation::sum_down_surface_mass(fine,coarse,mass_geometry(fd),mass_geometry(cd),rr);
#ifndef AMREX_USE_GPU
    EXPECT_EQ(std::fetestexcept(FE_OVERFLOW),0);
#endif
    expect_mass(coarse,[=] AMREX_GPU_HOST_DEVICE(int i,int,int k,bool valid) noexcept {
        return valid && k==0 ? (i==0 ? large : tiny) : Real(0);
    });
}
