#include "ERF.H"

using namespace amrex;

/**
 * @brief Group boxes by the connectivity of the pressure operator.
 * @param[in] bl List of boxes to partition.
 * @param[out] bins Vector of BoxArrays containing the grouped subdomains.
 */
void
ERF::make_subdomains (const BoxList& bl, Vector<BoxArray>& bins)
{
    Vector<BoxList> bins_bl;

    // Clear out any old bins
    bins.clear();

    // The regular constant-spacing Poisson stencil connects cell faces only.
    // Edge/corner contact leaves independent nullspaces and requires separate
    // solvability means. Keep the existing wider grouping for terrain and EB.
    const bool face_only = solverChoice.mesh_type == MeshType::ConstantDz &&
                           solverChoice.terrain_type != TerrainType::EB;
    const auto connected = [face_only] (const Box& left, const Box& right) {
        if (!face_only) {
            Box grown(left); grown.grow(1);
            return grown.intersects(right);
        }
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            Box grown(left); grown.grow(dir,1);
            if (grown.intersects(right)) { return true; }
        }
        return false;
    };

    // Iterate over boxes
    for (auto bx : bl)
    {
        // Find every existing bin this box connects. Stopping at the first match
        // is not enough: a box that bridges two bins connects them, so those bins
        // have to be merged.  Otherwise the two halves remain separate subdomains
        // even though they are now adjacent through bx, which breaks the invariant
        // that no box in a subdomain connects to a box in any other subdomain.
        Vector<int> touched;
        for (int j = 0; j < bins_bl.size(); ++j) {
            for (auto const& b : bins_bl[j])
            {
                if (connected(bx,b)) {
                    touched.push_back(j);
                    break;
                }
            }
        }

        if (touched.empty()) {
            // Box touches nothing seen so far, so it starts a new bin
            BoxList new_bin;
            new_bin.push_back(bx);
            bins_bl.push_back(new_bin);
        } else {
            // Keep the first bin touched and fold the others, plus bx, into it.
            // Erasing from the back leaves the lower indices -- including the one
            // we hold a reference to -- valid.
            BoxList& keep = bins_bl[touched[0]];
            keep.push_back(bx);
            for (int n = int(touched.size())-1; n >= 1; --n) {
                keep.join(bins_bl[touched[n]]);
                bins_bl.erase(bins_bl.begin() + touched[n]);
            }
        }
    }

#ifdef AMREX_DEBUG
    // Verify the invariant we just worked to maintain: boxes in different bins
    // must not connect, else the Poisson solve would treat one connected region as
    // two independent problems with bogus conditions at their shared interface.
    for (int i = 0; i < bins_bl.size(); ++i) {
        for (int j = i+1; j < bins_bl.size(); ++j) {
            for (auto const& bi : bins_bl[i]) {
                for (auto const& bj : bins_bl[j]) {
                    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!connected(bi,bj),
                        "make_subdomains: boxes in different subdomains are adjacent");
                }
            }
        }
    }
#endif

    // Convert the BoxLists to BoxArrays
    for (int i = 0; i < bins_bl.size(); ++i) {
        bins.push_back(BoxArray(bins_bl[i]));
    }
}
