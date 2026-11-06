#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_VisMF.H>
#include "ERF_IndexDefines.H"
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>
using namespace amrex;

namespace {
void require_finite (const MultiFab& field)
{
    if (field.contains_nan() || field.contains_inf()) {
        throw std::runtime_error("Checkpoint field contains nonfinite valid data");
    }
}
}

int main(int argc, char** argv)
{
    if (argc < 3) { return 2; }
    // Tool arguments are positional, not an AMReX input file.
    int amrex_argc = 1;
    Initialize(amrex_argc, argv);
    int code = 0;
    try {
        const std::string action(argv[1]);
        if (action == "seed") {
            if (argc != 5 && argc != 6) { throw std::runtime_error("seed requires checkpoint, history, rain mixing ratio, and optional level"); }
            const std::string checkpoint(argv[2]);
            const Real history = std::stod(argv[3]);
            const Real mixing = std::stod(argv[4]);
            const std::string level = argc > 5 ? argv[5] : "0";
            const std::string prefix = checkpoint + "/Level_" + level + "/";
            MultiFab cell, rain, baseline;
            VisMF::Read(cell, prefix + "Cell");
            VisMF::Read(rain, prefix + "RainAccum");
            VisMF::Read(baseline, prefix + "SlmPrecip");
            AMREX_ALWAYS_ASSERT(cell.nComp() > RhoQ3_comp);
            for (MFIter mfi(cell); mfi.isValid(); ++mfi) {
                const auto values = cell.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                    values(i,j,k,RhoQ3_comp) = mixing * values(i,j,k,Rho_comp);
                });
            }
            rain.setVal(0);
            for (MFIter mfi(rain); mfi.isValid(); ++mfi) {
                const auto values = rain.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                    if (k == 0) { values(i,j,k) = history; }
                });
            }
            baseline.setVal(0);
            baseline.setVal(history, 0, 1, 0);
            Gpu::streamSynchronize();
            VisMF::Write(cell, prefix + "Cell");
            VisMF::Write(rain, prefix + "RainAccum");
            VisMF::Write(baseline, prefix + "SlmPrecip");
            Print().SetPrecision(17) << "{\"action\":\"seed\",\"history\":" << history
                    << ",\"rain_mixing_ratio\":" << mixing << "}\n";
        } else if (action == "add-bottom-history") {
            if (argc != 4) { throw std::runtime_error("add-bottom-history requires a MultiFab prefix and finite offset"); }
            const Real offset = std::stod(argv[3]);
            if (!std::isfinite(offset)) { throw std::runtime_error("History offset must be finite"); }
            MultiFab field;
            VisMF::Read(field, argv[2]);
            require_finite(field);
            AMREX_ALWAYS_ASSERT(field.nComp() == 1);
            for (MFIter mfi(field); mfi.isValid(); ++mfi) {
                const auto values = field.array(mfi);
                ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                    if (k == 0) { values(i,j,k) += offset; }
                });
            }
            Gpu::streamSynchronize();
            require_finite(field);
            VisMF::Write(field, argv[2]);
            Print().SetPrecision(17) << "{\"action\":\"add-bottom-history\",\"offset\":" << offset << "}\n";
        } else if (action == "stats") {
            if (argc != 3) { throw std::runtime_error("stats requires a MultiFab prefix"); }
            MultiFab field;
            VisMF::Read(field, argv[2]);
            require_finite(field);
            for (int c = 0; c < field.nComp(); ++c) {
                Print().SetPrecision(17) << "{\"component\":" << c << ",\"min\":" << field.min(c)
                        << ",\"max\":" << field.max(c) << ",\"sum\":" << field.sum(c) << "}\n";
            }
        } else if (action == "compare") {
            if (argc != 5) { throw std::runtime_error("compare requires two MultiFab prefixes and the expected bottom-plane offset"); }
            MultiFab left, right;
            VisMF::Read(left, argv[2]);
            VisMF::Read(right, argv[3]);
            const Real offset = std::stod(argv[4]);
            AMREX_ALWAYS_ASSERT(left.boxArray() == right.boxArray());
            AMREX_ALWAYS_ASSERT(left.nComp() == right.nComp());
            require_finite(left);
            require_finite(right);
            Vector<Real> left_norm(left.nComp());
            Vector<Real> right_norm(right.nComp());
            for (int c = 0; c < left.nComp(); ++c) {
                left_norm[c] = left.norm0(c);
                right_norm[c] = right.norm0(c);
            }
            MultiFab::Subtract(right, left, 0, 0, left.nComp(), 0);
            for (MFIter mfi(right); mfi.isValid(); ++mfi) {
                const auto values = right.array(mfi);
                const int ncomp = right.nComp();
                ParallelFor(mfi.validbox(), ncomp,
                    [=] AMREX_GPU_DEVICE (int i,int j,int k,int n) noexcept {
                        if (k == 0) { values(i,j,k,n) -= offset; }
                    });
            }
            require_finite(right);
            for (int c = 0; c < right.nComp(); ++c) {
                Print().SetPrecision(17) << "{\"component\":" << c
                        << ",\"left_norm\":" << left_norm[c]
                        << ",\"right_norm\":" << right_norm[c]
                        << ",\"max_abs_diff\":" << right.norm0(c) << "}\n";
            }
        } else if (action == "baseline") {
            if (argc != 4) { throw std::runtime_error("baseline requires checkpoint and level (Kessler units)"); }
            const std::string prefix = std::string(argv[2]) + "/Level_" + argv[3] + "/";
            MultiFab rain, baseline;
            VisMF::Read(rain, prefix + "RainAccum");
            VisMF::Read(baseline, prefix + "SlmPrecip");
            require_finite(rain);
            require_finite(baseline);
            AMREX_ALWAYS_ASSERT(rain.size() == baseline.size());
            AMREX_ALWAYS_ASSERT(rain.DistributionMap() == baseline.DistributionMap());
            MultiFab difference(baseline.boxArray(), baseline.DistributionMap(), 1, 0);
            difference.setVal(0);
            Long surface_cells = 0;
            for (MFIter mfi(baseline); mfi.isValid(); ++mfi) {
                const Box& atmospheric_box = rain.boxArray()[mfi.index()];
                if (atmospheric_box.smallEnd(2) != 0) { continue; }
                const auto native = rain.const_array(mfi);
                const auto stored = baseline.const_array(mfi);
                const auto error = difference.array(mfi);
                const Box surface_box = mfi.validbox();
                AMREX_ALWAYS_ASSERT(surface_box.smallEnd(2) == 0 && surface_box.bigEnd(2) == 0);
                surface_cells += surface_box.numPts();
                ParallelFor(surface_box, [=] AMREX_GPU_DEVICE (int i,int j,int k) noexcept {
                    error(i,j,k) = stored(i,j,k) - native(i,j,k);
                });
            }
            ParallelDescriptor::ReduceLongSum(surface_cells);
            require_finite(difference);
            Print().SetPrecision(17) << "{\"max_abs_diff\":" << difference.norm0()
                    << ",\"surface_cells\":" << surface_cells << "}\n";
        } else {
            code = 2;
        }
    } catch (const std::exception& error) {
        Print() << error.what() << '\n';
        code = 1;
    }
    Finalize();
    return code;
}
