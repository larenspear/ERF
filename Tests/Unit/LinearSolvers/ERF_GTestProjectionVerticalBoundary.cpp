#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MultiFab.H>
#ifdef ERF_USE_FFT
#include <AMReX_FFT_Poisson.H>
#endif
#include "ERF_SolverUtils.H"
#include <cmath>
#include <limits>
#include <gtest/gtest.h>

using namespace amrex;

namespace {

void check_singularity ()
{
    const Box domain(IntVect(0),IntVect(7));
    const RealBox physical(0,0,0,8,8,8);
    Geometry geom(domain,physical,0,{0,0,0});
    Array<std::string,6> names;
    names.fill("SlipWall");
    EXPECT_TRUE(projection_is_singular(geom,names,domain,false));
    for (int direction=0; direction<3; ++direction) {
        for (auto side : {Orientation::low,Orientation::high}) {
            const Orientation face(direction,side);
            for (const char* type : {"Outflow","Open"}) {
                names.fill("SlipWall"); names[face]=type;
                EXPECT_FALSE(projection_is_singular(geom,names,domain,false));
                EXPECT_EQ(projection_is_singular(geom,names,domain,true),direction != 2);
                Box interior=domain;
                if (side==Orientation::low) interior.setSmall(direction,2);
                else interior.setBig(direction,5);
                EXPECT_TRUE(projection_is_singular(geom,names,interior,false));
                Array<int,3> periodic{0,0,0}; periodic[direction]=1;
                Geometry periodic_geom(domain,physical,0,periodic);
                EXPECT_TRUE(projection_is_singular(periodic_geom,names,domain,false));
            }
        }
    }
}

// Exact-sized allocations catch the nz and -1 endpoint reads under sanitizers.
// The device kernel makes the same helper compile in every enabled GPU backend.
void check_vertical_spacing ()
{
    for (int lo : {0,2}) {
        for (int cells : {1,3,8}) {
            Vector<Real> host(lo+cells,Real(-100));
            for (int k=lo;k<lo+cells;++k) host[k]=Real(k-lo+1);
            Gpu::DeviceVector<Real> dz(host.size()), distance(cells+1);
            Gpu::copy(Gpu::hostToDevice,host.begin(),host.end(),dz.begin());
            const Real* ptr=dz.data(); Real* result=distance.data();
            ParallelFor(cells+1,[=] AMREX_GPU_DEVICE (int n) noexcept {
                result[n]=projection_vertical_spacing(lo+n,lo,lo+cells-1,ptr);
            });
            Vector<Real> actual(cells+1);
            Gpu::copy(Gpu::deviceToHost,distance.begin(),distance.end(),actual.begin());
            EXPECT_EQ(actual.front(),Real(1));
            EXPECT_EQ(actual.back(),Real(cells));
            for (int n=1;n<cells;++n) EXPECT_EQ(actual[n],Real(n)+Real(0.5));
        }
    }
}

#ifdef ERF_USE_FFT
// Supply an independently assembled finite-volume RHS and check the returned
// pressure's flux divergence, including both physical boundary rows. The input
// profile is arbitrary; odd/even ghost extension supplies the boundary data.
void check_vertical_balance ()
{
    constexpr int nz=8;
    const Box domain(IntVect(0),IntVect(3,3,nz-1));
    const RealBox physical(0,0,0,4,4,nz);
    Geometry geom(domain,physical,0,{1,1,0});
    BoxArray boxes(domain); boxes.maxSize(4);
    DistributionMapping mapping(boxes);
    Array<std::string,6> names; names.fill("Periodic");
    for (bool stretched : {false,true}) {
        Vector<Real> heights(nz), exact(nz);
        for (int k=0;k<nz;++k) {
            heights[k]=stretched ? Real(1)+Real(0.125)*k : Real(1);
            exact[k]=std::sin(Real(0.7)*k+Real(0.2));
        }
        Gpu::DeviceVector<Real> dz(nz);
        Gpu::copy(Gpu::hostToDevice,heights.begin(),heights.end(),dz.begin());
        const Real* dzptr=dz.data();
        for (bool lower_dirichlet : {false,true}) {
            for (bool upper_dirichlet : {false,true}) {
                names[2]=lower_dirichlet ? "Outflow" : "SlipWall";
                names[5]=upper_dirichlet ? "Outflow" : "SlipWall";
                const auto bc=get_fft_bc(geom,names,domain,false);
                Vector<Real> source(nz);
                for (int k=0;k<nz;++k) {
                    const Real below=k ? exact[k-1] : (lower_dirichlet ? -exact[0] : exact[0]);
                    const Real above=k+1<nz ? exact[k+1] : (upper_dirichlet ? -exact[nz-1] : exact[nz-1]);
                    const Real lower_distance=k ? (heights[k]+heights[k-1])/Real(2) : heights[0];
                    const Real upper_distance=k+1<nz ? (heights[k]+heights[k+1])/Real(2) : heights[nz-1];
                    source[k]=((above-exact[k])/upper_distance-(exact[k]-below)/lower_distance)/heights[k];
                }
                Gpu::DeviceVector<Real> data(nz);
                Gpu::copy(Gpu::hostToDevice,source.begin(),source.end(),data.begin());
                const Real* sourceptr=data.data();
                MultiFab rhs(boxes,mapping,1,0),phi(boxes,mapping,1,1),residual(boxes,mapping,1,0);
                for (MFIter mfi(rhs);mfi.isValid();++mfi) {
                    const auto r=rhs.array(mfi);
                    ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {r(i,j,k)=sourceptr[k];});
                }
                phi.setVal(Real(0));
                if (stretched) { FFT::PoissonHybrid<MultiFab> solver(geom,bc); solver.solve(phi,rhs,dz); }
                else { FFT::Poisson<MultiFab> solver(geom,bc); solver.solve(phi,rhs); }
                phi.FillBoundary(geom.periodicity());
                for (MFIter mfi(phi);mfi.isValid();++mfi) {
                    const Box box=mfi.validbox(); const auto p=phi.array(mfi);
                    ParallelFor(box,[=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                        if (k==0) p(i,j,-1)=lower_dirichlet ? -p(i,j,0) : p(i,j,0);
                        if (k==nz-1) p(i,j,nz)=upper_dirichlet ? -p(i,j,nz-1) : p(i,j,nz-1);
                    });
                }
                for (MFIter mfi(residual);mfi.isValid();++mfi) {
                    const auto p=phi.const_array(mfi); const auto r=residual.array(mfi);
                    ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                        const Real below=-(p(i,j,k)-p(i,j,k-1))/projection_vertical_spacing(k,0,nz-1,dzptr);
                        const Real above=-(p(i,j,k+1)-p(i,j,k))/projection_vertical_spacing(k+1,0,nz-1,dzptr);
                        r(i,j,k)=sourceptr[k]+(above-below)/dzptr[k];
                    });
                }
                const Real error=residual.norm0();
                EXPECT_TRUE(std::isfinite(error));
                // Small direct FFT/tridiagonal solves: roundoff, not discretization error.
                EXPECT_LE(error,Real(1024)*std::numeric_limits<Real>::epsilon()*rhs.norm0())
                    << stretched << ' ' << lower_dirichlet << ' ' << upper_dirichlet;
            }
        }
    }
}
#endif

} // namespace

TEST(ProjectionVerticalBoundary, SingularityFollowsEveryPhysicalFace)
{
    check_singularity();
}

TEST(ProjectionVerticalBoundary, SpacingUsesOnlyAdjacentStoredCells)
{
    check_vertical_spacing();
}

#ifdef ERF_USE_FFT
TEST(ProjectionVerticalBoundary, PoissonFluxBalancesAllVerticalBoundaryPairs)
{
    check_vertical_balance();
}

#endif
