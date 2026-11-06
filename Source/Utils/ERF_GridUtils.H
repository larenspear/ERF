#ifndef ERF_GRID_UTILS_H_
#define ERF_GRID_UTILS_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>

#include <AMReX.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

namespace erf_grid_utils {

struct UniformGridMetadata
{
    int nx{0};
    int ny{0};
    amrex::Real dx{amrex::Real(0.0)};
    amrex::Real dy{amrex::Real(0.0)};
    amrex::Real xmin{amrex::Real(0.0)};
    amrex::Real ymin{amrex::Real(0.0)};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
comparison_tolerance (amrex::Real lhs, amrex::Real rhs) noexcept
{
    amrex::Real scale = amrex::Real(1.0);
    const amrex::Real lhs_abs = amrex::Math::abs(lhs);
    const amrex::Real rhs_abs = amrex::Math::abs(rhs);
    if (lhs_abs > scale) scale = lhs_abs;
    if (rhs_abs > scale) scale = rhs_abs;
    // NetCDF coordinate variables are commonly stored as float32 even when
    // ERF is built with double precision.  A few float32 ulps at the
    // coordinate magnitude are therefore source-file quantization, not grid
    // nonuniformity.  The Real tolerance is retained for native Real data;
    // the float tolerance is deliberately a small ulp-based allowance rather
    // than an arbitrary absolute threshold.
    const amrex::Real real_ulp =
        amrex::Real(64.0) * std::numeric_limits<amrex::Real>::epsilon() * scale;
    const amrex::Real float_ulp =
        amrex::Real(8.0) * static_cast<amrex::Real>(std::numeric_limits<float>::epsilon()) * scale;
    return (real_ulp > float_ulp) ? real_ulp : float_ulp;
}

inline bool
nearly_equal (amrex::Real lhs, amrex::Real rhs)
{
    return std::abs(lhs - rhs) <= comparison_tolerance(lhs, rhs);
}

inline std::string
validate_uniform_axis (const amrex::Vector<amrex::Real>& coordinates,
                       int expected_size,
                       const std::string& axis_name,
                       const std::string& field_description,
                       amrex::Real& origin,
                       amrex::Real& spacing)
{
    if (expected_size < 2) {
        return field_description + " requires at least two " + axis_name +
               " coordinates for bilinear interpolation";
    }
    if (coordinates.size() != static_cast<amrex::Long>(expected_size)) {
        return field_description + " has " + axis_name + " coordinate length " +
               std::to_string(coordinates.size()) + ", expected " +
               std::to_string(expected_size);
    }

    origin = coordinates.front();
    // Derive the spacing from the two endpoints rather than from the first
    // interval alone.  The returned origin/spacing pair is what
    // uniform_interpolation_stencil uses to reconstruct the axis as
    // origin + index*spacing, so a first interval that is rounded slightly
    // short or long would otherwise accumulate that error over the whole axis
    // -- for a float32 axis with many points the reconstructed far end can
    // drift by (point_count-1) ulps, which is large enough to shift the
    // stencil by a cell near the upper edge.  Averaging over the full extent
    // pins both endpoints instead.  A non-finite last coordinate propagates
    // into spacing and is caught by the isfinite check below.
    spacing = (coordinates[expected_size-1] - coordinates.front()) /
              static_cast<amrex::Real>(expected_size - 1);
    if (!std::isfinite(origin) || !std::isfinite(spacing) || spacing <= amrex::Real(0.0)) {
        return field_description + " must have finite, strictly increasing " +
               axis_name + " coordinates with nonzero spacing";
    }

    for (int index = 1; index < expected_size; ++index) {
        const amrex::Real coordinate = coordinates[index];
        if (!std::isfinite(coordinate) || coordinate <= coordinates[index-1]) {
            return field_description + " must have finite, strictly increasing " +
                   axis_name + " coordinates";
        }
        // Compare each interval against the mean spacing instead of
        // accumulating the first spacing over the whole axis.  This avoids
        // rejecting a valid float32 axis because each stored coordinate was
        // rounded independently, while still rejecting materially nonuniform
        // grids.
        //
        // The tolerance is the *larger* of the two allowances, not the
        // smaller: the ulp term covers source-file quantization, which scales
        // with the coordinate magnitude, while the fraction-of-spacing term
        // covers grids whose coordinates are small enough that a few ulps are
        // negligible.  Taking the minimum cancels the ulp allowance in exactly
        // the case it exists for -- a fine grid at a large offset.  Concretely,
        // at a UTM easting near 5e5 the float32 ulp is 0.03125 m, so a 0.4 m
        // raster stores intervals that deviate from the nominal spacing by up
        // to 0.025 m: more than 5% of the spacing, yet far below the ulp
        // allowance.  Integer-metre spacings are unaffected either way, since
        // integers are exactly representable in float32 at UTM magnitudes.
        const amrex::Real local_spacing = coordinate - coordinates[index-1];
        const amrex::Real spacing_tolerance = std::max(
            amrex::Real(0.05) * std::abs(spacing),
            comparison_tolerance(coordinate, coordinates[index-1]));
        if (std::abs(local_spacing - spacing) > spacing_tolerance) {
            return field_description + " has nonuniform " + axis_name +
                   " coordinates, which are not supported";
        }
    }

    return {};
}

inline std::string
validate_matching_grid (const UniformGridMetadata& reference,
                        const UniformGridMetadata& candidate,
                        const std::string& reference_description,
                        const std::string& candidate_description)
{
    if (candidate.nx != reference.nx || candidate.ny != reference.ny) {
        return candidate_description + " dimensions (" + std::to_string(candidate.nx) +
               " x " + std::to_string(candidate.ny) + ") do not match " +
               reference_description + " dimensions (" + std::to_string(reference.nx) +
               " x " + std::to_string(reference.ny) + ")";
    }

    struct MetadataValue {
        const char* name;
        amrex::Real reference_value;
        amrex::Real candidate_value;
    };
    const MetadataValue values[] = {
        {"x spacing", reference.dx, candidate.dx},
        {"y spacing", reference.dy, candidate.dy},
        {"x origin", reference.xmin, candidate.xmin},
        {"y origin", reference.ymin, candidate.ymin}
    };
    for (const auto& value : values) {
        if (!nearly_equal(value.reference_value, value.candidate_value)) {
            std::ostringstream message;
            message << candidate_description << " " << value.name << " ("
                    << value.candidate_value << ") does not match "
                    << reference_description << " (" << value.reference_value << ")";
            return message.str();
        }
    }

    return {};
}

/**
 * \brief A geometrically stretched vertical grid: dz_k = dz0 * ratio^k.
 */
struct GeometricStretch
{
    amrex::Real dz0{amrex::Real(0.0)};      //!< thickness of the first layer, as actually used
    amrex::Real ratio{amrex::Real(1.0)};    //!< layer-to-layer stretch factor
    amrex::Real residual{amrex::Real(0.0)}; //!< evaluated sum of the layers minus z_top
    bool uniform{false};                 //!< true when the requested dz0 could not be honored
    bool converged{false};               //!< tolerance met or storage-limited ratio selected
};

// Horner's rule avoids cancellation in (r^nz-1)/(r-1) near r=1. Overflow is
// permitted while bracketing: an infinite positive depth is an upper bound.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
geometric_depth (amrex::Real dz0, amrex::Real ratio, int nz) noexcept
{
    amrex::Real depth = dz0;
    for (int k = 1; k < nz; ++k) {
        depth = depth * ratio + dz0;
    }
    return depth;
}

/**
 * \brief Find the geometric ratio that fills a column with nz growing layers.
 *
 * The polynomial depth is monotone for ratio >= 1. Bracket its root, then bisect
 * until the depth tolerance is met or the bracket has adjacent representable
 * endpoints. In the latter case choose the endpoint with smaller evaluated
 * residual. The lower bound r=1 is selected if rounding already puts its depth
 * at or above z_top. The requested absolute tolerance can be smaller than one ratio ULP
 * permits, especially in single precision. residual always reports that error.
 * The tolerance is capped at half the first thickness: a tolerance larger than
 * the column must not accept a grid whose forced last node inverts a layer.
 * build_geometric_zlevels also checks the stored nodes after rounding.
 *
 * If dz0 is at least the uniform spacing, or nz == 1, return a uniform grid.
 * Invalid/nonfinite inputs, an unrepresentable root, or exhausted iteration
 * bounds return converged == false. No convergence rate is assumed.
 *
 * @param[in] dz0   requested thickness of the first layer
 * @param[in] z_top positive finite column depth
 * @param[in] nz    number of layers
 * @param[in] tol   positive finite absolute depth tolerance
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GeometricStretch
geometric_stretch (amrex::Real dz0,
                   amrex::Real z_top,
                   int nz,
                   amrex::Real tol) noexcept
{
    GeometricStretch gs;
    const amrex::Real largest = std::numeric_limits<amrex::Real>::max();
    const amrex::Real Nz = static_cast<amrex::Real>(nz);
    if (nz < 1 || !(z_top > amrex::Real(0.0)) || !(dz0 > amrex::Real(0.0)) ||
        !(tol > amrex::Real(0.0)) || !(z_top <= largest) || !(dz0 <= largest) ||
        !(tol <= largest)) {
        return gs;
    }
    if (nz == 1 || dz0 >= z_top/Nz) {
        gs.dz0       = z_top/Nz;
        gs.ratio     = amrex::Real(1.0);
        gs.residual  = amrex::Real(0.0);
        gs.uniform   = true;
        gs.converged = (gs.dz0 > amrex::Real(0.0));
        return gs;
    }

    const amrex::Real target = amrex::min(tol, dz0/amrex::Real(2.0));
    amrex::Real lower = amrex::Real(1.0);
    amrex::Real upper = amrex::Real(2.0);
    amrex::Real lower_residual = geometric_depth(dz0, lower, nz) - z_top;
    // A nearly uniform request can round above the top even at the smallest
    // allowed ratio. No representable ratio >= 1 improves that evaluated sum.
    if (lower_residual >= amrex::Real(0.0)) {
        gs.dz0       = dz0;
        gs.ratio     = lower;
        gs.residual  = lower_residual;
        gs.uniform   = false;
        gs.converged = (amrex::Math::abs(lower_residual) <= largest);
        return gs;
    }
    amrex::Real upper_residual = geometric_depth(dz0, upper, nz) - z_top;
    // Doubling reaches the largest finite ratio in at most max_exponent steps.
    const int bracket_limit = std::numeric_limits<amrex::Real>::max_exponent;
    for (int iter = 0; upper_residual < amrex::Real(0.0) && iter < bracket_limit; ++iter) {
        lower = upper;
        lower_residual = upper_residual;
        upper = (upper <= largest/amrex::Real(2.0)) ? upper*amrex::Real(2.0) : largest;
        upper_residual = geometric_depth(dz0, upper, nz) - z_top;
    }
    if (!(lower_residual <= amrex::Real(0.0)) || !(upper_residual >= amrex::Real(0.0))) {
        return gs;
    }

    // This bound covers the exponent range and the significand bits, including
    // a bracket spanning the whole positive finite range. Rounding can finish
    // the solve earlier when no representable midpoint remains.
    const int solve_limit = std::numeric_limits<amrex::Real>::max_exponent +
                            std::numeric_limits<amrex::Real>::digits;
    for (int iter = 0; iter < solve_limit; ++iter) {
        amrex::Real ratio = lower + (upper-lower)/amrex::Real(2.0);
        const bool adjacent = (ratio == lower || ratio == upper);
        if (adjacent) {
            ratio = (amrex::Math::abs(lower_residual) <= amrex::Math::abs(upper_residual)) ? lower : upper;
        }
        const amrex::Real residual = geometric_depth(dz0, ratio, nz) - z_top;
        if (amrex::Math::abs(residual) <= target || adjacent) {
            gs.dz0       = dz0;
            gs.ratio     = ratio;
            gs.residual  = residual;
            gs.uniform   = false;
            gs.converged = (amrex::Math::abs(residual) <= largest);
            return gs;
        }
        if (residual < amrex::Real(0.0)) {
            lower = ratio;
            lower_residual = residual;
        } else {
            upper = ratio;
            upper_residual = residual;
        }
    }
    return gs;
}

/**
 * \brief Build geometric nodes and check their stored, rounded monotonicity.
 *
 * zlevels is sized nz+1 on entry. The final node is assigned z_top exactly.
 * A failed solve leaves the array unchanged. If construction or the forced top
 * produces a nonfinite or nonincreasing node, return converged == false; callers
 * must check this flag before consuming the array.
 */
inline GeometricStretch
build_geometric_zlevels (amrex::Vector<amrex::Real>& zlevels,
                         amrex::Real dz0,
                         amrex::Real z_top,
                         amrex::Real tol)
{
    const int nz = static_cast<int>(zlevels.size()) - 1;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nz >= 1,
        "build_geometric_zlevels needs a zlevels array of at least two entries");

    GeometricStretch gs = geometric_stretch(dz0, z_top, nz, tol);
    if (!gs.converged) { return gs; }
    amrex::Real dz = gs.dz0;
    zlevels[0] = amrex::Real(0.0);
    zlevels[1] = dz;
    // Compute each later thickness only when its layer will be stored. An
    // unused multiplication after the final layer can overflow a valid grid.
    for (int k(2); k<=nz; ++k) {
        dz *= gs.ratio;
        zlevels[k] = zlevels[k-1] + dz;
    }
    zlevels[nz] = z_top;
    for (int k = 1; k <= nz; ++k) {
        if (!(zlevels[k] > zlevels[k-1]) ||
            !(zlevels[k] <= std::numeric_limits<amrex::Real>::max())) {
            gs.converged = false;
        }
    }
    return gs;
}

struct InterpolationStencil
{
    int lower{0};
    amrex::Real weight{amrex::Real(0.0)};
    bool inside{false};
};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
InterpolationStencil
uniform_interpolation_stencil (amrex::Real coordinate,
                               amrex::Real origin,
                               amrex::Real spacing,
                               int point_count) noexcept
{
    InterpolationStencil stencil;
    if (point_count < 2 || spacing <= amrex::Real(0.0)) {
        return stencil;
    }

    const amrex::Real upper = origin + static_cast<amrex::Real>(point_count - 1) * spacing;
    const amrex::Real lower_tolerance = comparison_tolerance(coordinate, origin);
    const amrex::Real upper_tolerance = comparison_tolerance(coordinate, upper);

    if (coordinate < origin - lower_tolerance || coordinate > upper + upper_tolerance) {
        return stencil;
    }

    const amrex::Real bounded_coordinate =
        (coordinate <= origin + lower_tolerance) ? origin :
        ((coordinate >= upper - upper_tolerance) ? upper : coordinate);
    const amrex::Real fractional_index = (bounded_coordinate - origin) / spacing;
    int lower = static_cast<int>(amrex::Math::floor(fractional_index));
    lower = (lower < point_count - 1) ? lower : point_count - 2;

    amrex::Real weight = fractional_index - static_cast<amrex::Real>(lower);
    weight = (weight < amrex::Real(0.0)) ? amrex::Real(0.0) :
             ((weight > amrex::Real(1.0)) ? amrex::Real(1.0) : weight);

    stencil.lower = lower;
    stencil.weight = weight;
    stencil.inside = true;
    return stencil;
}

} // namespace erf_grid_utils

#endif
