#include <AMReX.H>
#include <AMReX_Geometry.H>
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

Geometry geometry (Box const& box, Array<int,3> const& periodic)
{
    RealBox rb(0, 0, 0, box.length(0), box.length(1), box.length(2));
    return Geometry(box, rb, 0, periodic);
}

Box interval (int direction, int lo, int hi)
{
    Box box(IntVect(0), IntVect(7));
    box.setSmall(direction, lo);
    box.setBig(direction, hi);
    return box;
}

#ifdef ERF_USE_FFT
// Keep device lambdas outside the protected GoogleTest TestBody member so
// nvcc can instantiate their extended device closures.
void fill_mode_rhs (MultiFab& rhs, Real wave, Real eigenvalue, int direction, int lower)
{
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        auto const a = rhs.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
            const int position = direction == 0 ? i : j;
            a(i,j,k) = eigenvalue*std::cos(wave*(Real(position-lower)+Real(0.5)));
        });
    }
}

void compute_mode_error (MultiFab& error, MultiFab const& phi,
                         Real wave, int direction, int lower)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto const p = phi.const_array(mfi);
        auto const e = error.array(mfi);
        ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
            const int position = direction == 0 ? i : j;
            e(i,j,k) = p(i,j,k)-std::cos(wave*(Real(position-lower)+Real(0.5)));
        });
    }
}

#endif

} // namespace

// Exhaust all disjoint pairs of intervals in each axis, including both input
// orders. The independent oracle is face sharing on an eight-cell circle.
TEST(ProjectionTopology, FindsPeriodicFaceConnections)
{
    const Box domain(IntVect(0), IntVect(7));
    for (int direction = 0; direction < 3; ++direction) {
        Array<int,3> periodic{0,0,0};
        periodic[direction] = 1;
        const auto geom = geometry(domain, periodic);
        for (int alo = 0; alo < 8; ++alo) {
        for (int ahi = alo; ahi < 8; ++ahi) {
        for (int blo = 0; blo < 8; ++blo) {
        for (int bhi = blo; bhi < 8; ++bhi) {
            if (!(ahi < blo || bhi < alo)) { continue; }
            Vector<BoxArray> bins{BoxArray(interval(direction,alo,ahi)),
                                  BoxArray(interval(direction,blo,bhi))};
            const bool connected = (alo == 0 && bhi == 7) || (blo == 0 && ahi == 7);
            EXPECT_EQ(has_periodic_subdomain_connections(geom,bins), connected)
                << direction << ": [" << alo << ',' << ahi << "] [" << blo << ',' << bhi << ']';
        }}}}
    }
}

TEST(ProjectionTopology, DistinguishesSelfSeamsCoarseNeighborsAndCornerContact)
{
    const Box domain(IntVect(0), IntVect(7));
    const auto periodic = geometry(domain, {1,1,0});
    const auto nonperiodic = geometry(domain, {0,0,0});
    Vector<BoxArray> bins{BoxArray(domain)};
    EXPECT_FALSE(has_periodic_subdomain_connections(periodic,bins));
    bins = {BoxArray(interval(0,0,2))};
    EXPECT_FALSE(has_periodic_subdomain_connections(periodic,bins));
    bins.push_back(BoxArray(interval(0,5,7)));
    EXPECT_TRUE(has_periodic_subdomain_connections(periodic,bins));
    EXPECT_FALSE(has_periodic_subdomain_connections(nonperiodic,bins));

    // Periodic corner contact alone does not share a nearest-neighbor face.
    bins = {BoxArray(Box(IntVect(0,0,0),IntVect(2,2,7))),
            BoxArray(Box(IntVect(5,5,0),IntVect(7,7,7)))};
    EXPECT_FALSE(has_periodic_subdomain_connections(periodic,bins));

    const IntVect offset(-4,3,5);
    Box translated_domain = domain;
    translated_domain.shift(offset);
    const auto translated = geometry(translated_domain,{1,0,0});
    Box translated_low = interval(0,0,2);
    Box translated_high = interval(0,5,7);
    translated_low.shift(offset);
    translated_high.shift(offset);
    bins = {BoxArray(translated_low),BoxArray(translated_high)};
    EXPECT_TRUE(has_periodic_subdomain_connections(translated,bins));

    // Actual boxes, not overlapping bounding boxes, determine connectivity.
    BoxList sparse;
    sparse.push_back(Box(IntVect(5,0,0),IntVect(7,1,7)));
    sparse.push_back(Box(IntVect(5,6,0),IntVect(7,7,7)));
    bins = {BoxArray(Box(IntVect(0,3,0),IntVect(2,4,7))), BoxArray(sparse)};
    EXPECT_FALSE(has_periodic_subdomain_connections(periodic,bins));
    sparse.push_back(Box(IntVect(5,4,0),IntVect(7,4,7)));
    bins[1] = BoxArray(sparse);
    EXPECT_TRUE(has_periodic_subdomain_connections(periodic,bins));
}

#ifdef ERF_USE_FFT

TEST(FFTBoundary, IsolatedPartialPeriodicBoxesUseCoarseFineNeumannFaces)
{
    const Box domain(IntVect(0),IntVect(7));
    const auto geom = geometry(domain,{1,1,0});
    Array<std::string,6> names;
    names.fill("Open"); // Periodicity overrides inconsistent physical names too.
    for (int direction = 0; direction < 2; ++direction) {
        for (auto const& limits : {std::pair{0,3}, std::pair{4,7}, std::pair{2,5}}) {
            const auto bc = get_fft_bc(geom,names,interval(direction,limits.first,limits.second),false);
            EXPECT_EQ(bc[direction].first,FFT::Boundary::even);
            EXPECT_EQ(bc[direction].second,FFT::Boundary::even);
            EXPECT_EQ(bc[1-direction].first,FFT::Boundary::periodic);
            EXPECT_EQ(bc[1-direction].second,FFT::Boundary::periodic);
        }
    }
}

TEST(FFTBoundary, KeepsPhysicalOddFacesAndRealLateralNeumannFaces)
{
    const Box domain(IntVect(0),IntVect(7));
    const auto geom = geometry(domain,{0,0,0});
    Array<std::string,6> names;
    names.fill("Outflow");
    const auto ordinary = get_fft_bc(geom,names,domain,false);
    const auto real = get_fft_bc(geom,names,domain,true);
    for (int direction = 0; direction < 3; ++direction) {
        EXPECT_EQ(ordinary[direction].first,FFT::Boundary::odd);
        EXPECT_EQ(ordinary[direction].second,FFT::Boundary::odd);
        EXPECT_EQ(real[direction].first,direction == 2 ? FFT::Boundary::odd : FFT::Boundary::even);
        EXPECT_EQ(real[direction].second,direction == 2 ? FFT::Boundary::odd : FFT::Boundary::even);
    }
}

// The cell-centered cosine is an exact eigenvector of the discrete Neumann
// Laplacian, including its boundary rows. Both actual AMReX solvers must recover
// it, so merely constructing a solver with admissible labels is insufficient.
TEST(FFTBoundary, PartialPeriodicSolversRecoverDiscreteNeumannEigenmode)
{
    const Box domain(IntVect(0),IntVect(15,15,7));
    const auto full = geometry(domain,{1,1,0});
    Array<std::string,6> names;
    names.fill("SlipWall");
    names[0] = names[1] = names[3] = names[4] = "Periodic";
    for (int direction = 0; direction < 2; ++direction) {
        for (auto const& limits : {std::pair{0,7}, std::pair{4,11},
                                  std::pair{8,15}, std::pair{0,15}}) {
            const int lower = limits.first;
            const int width = limits.second-lower+1;
            const bool full_axis = width == domain.length(direction);
            Box region = domain;
            region.setSmall(direction,lower);
            region.setBig(direction,limits.second);
            Array<int,3> periodic{1,1,0};
            periodic[direction] = full_axis ? 1 : 0;
            const auto local = geometry(region,periodic);
            BoxArray ba(region);
            ba.maxSize(4); // Exercise internal fine-fine interfaces as well.
            DistributionMapping dm(ba);
            MultiFab rhs(ba,dm,1,0), phi(ba,dm,1,1), error(ba,dm,1,0);
            const Real wave = (full_axis ? Real(2) : Real(1))*std::acos(Real(-1))/Real(width);
            const Real eigenvalue = Real(2)*(std::cos(wave)-Real(1));
            fill_mode_rhs(rhs, wave, eigenvalue, direction, lower);
            const auto bc = get_fft_bc(full,names,region,false);
            FFT::Poisson<MultiFab> poisson(local,bc);
            FFT::PoissonHybrid<MultiFab> hybrid(local,bc);
            Vector<Real> dz_host(region.length(2),Real(1));
            Gpu::DeviceVector<Real> dz(dz_host.size());
            Gpu::copy(Gpu::hostToDevice,dz_host.begin(),dz_host.end(),dz.begin());
            for (bool use_hybrid : {false,true}) {
                phi.setVal(Real(0));
                if (use_hybrid) { hybrid.solve(phi,rhs,dz); }
                else { poisson.solve(phi,rhs); }
                ASSERT_FALSE(phi.contains_nan(0,1,0));
                ASSERT_FALSE(phi.contains_inf(0,1,0));
                compute_mode_error(error, phi, wave, direction, lower);
                const Real max_error = error.norm0();
                EXPECT_TRUE(std::isfinite(max_error));
                // Small FFT roundoff, amplified by 1/|lambda| < 7, fits within
                // this budget of 1024 epsilon for a unit-amplitude solution.
                EXPECT_LE(max_error, Real(1024)*std::numeric_limits<Real>::epsilon())
                    << direction << ' ' << lower << " hybrid=" << use_hybrid;
            }
        }
    }
}
#endif
