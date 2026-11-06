#ifndef ERF_GROUND_MASS_CHECKPOINT_H_
#define ERF_GROUND_MASS_CHECKPOINT_H_

#include "ERF_GroundMass.H"
#include <AMReX_Utility.H>
#include <AMReX_ParallelDescriptor.H>
#include <fstream>
#include <set>
#include <sstream>

namespace erf_ground_mass {

// Narrow compatibility reader for recent AMReX v2.1 particle headers. Check
// the whole header, including the grid table, before using its component names
// to establish that a missing ground manifest describes an empty field set.
inline bool matches_empty_particle_header (
    const std::string& text, const amrex::Vector<std::string>& expected_real_names)
{
    if (expected_real_names.empty()) return false;
    std::istringstream input(text);
    std::string version, name;
    int dimension = 0;
    long long count = 0;
    if (!(input >> version >> dimension >> count) ||
        (version != "Version_Two_Dot_One_double" && version != "Version_Two_Dot_One_single") ||
        dimension != AMREX_SPACEDIM || count != static_cast<long long>(expected_real_names.size())) {
        return false;
    }
    for (const auto& expected : expected_real_names) {
        if (!(input >> name) || name != expected) return false;
    }
    // Bound allocations and loops by the input length before parsing the rest.
    if (!(input >> count) || count < 0 || static_cast<unsigned long long>(count) > text.size()) return false;
    for (long long n = 0; n < count; ++n) {
        if (!(input >> name)) return false;
    }
    int is_checkpoint = 0;
    long long num_particles = 0, next_id = 0, finest = 0;
    if (!(input >> is_checkpoint >> num_particles >> next_id >> finest) ||
        is_checkpoint != 1 || num_particles < 0 || next_id <= 0 || finest < 0 ||
        static_cast<unsigned long long>(finest) >= text.size()) return false;
    amrex::Vector<long long> grids(static_cast<std::size_t>(finest)+1);
    for (auto& number : grids) {
        if (!(input >> number) || number <= 0 || static_cast<unsigned long long>(number) > text.size()) return false;
    }
    long long total_particles = 0;
    for (const auto number : grids) {
        for (long long n = 0; n < number; ++n) {
            long long which = 0, particles = 0, where = 0;
            if (!(input >> which >> particles >> where) || which < 0 || particles < 0 || where < 0 ||
                particles > num_particles-total_particles) return false;
            total_particles += particles;
        }
    }
    input >> std::ws;
    return input.eof() && total_particles == num_particles;
}

inline bool legacy_empty_is_compatible (
    const std::string& particle_header, const amrex::Vector<std::string>& expected_real_names)
{
    if (!amrex::FileExists(particle_header)) return false;
    amrex::Vector<char> data;
    amrex::ParallelDescriptor::ReadAndBcastFile(particle_header, data);
    return matches_empty_particle_header(std::string(data.dataPtr(), data.size()-1), expected_real_names);
}

inline std::string manifest (const GroundMassAccumulationFields& fields)
{
    std::string text = "ERF ground mass v1\n" + std::to_string(fields.size()) + "\n";
    std::set<std::string> names;
    for (const auto& field : fields) {
        AMREX_ALWAYS_ASSERT(field.mass != nullptr);
        AMREX_ALWAYS_ASSERT(!field.name.empty());
        for (const char c : field.name) {
            AMREX_ALWAYS_ASSERT((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                (c >= '0' && c <= '9') || c == '_');
        }
        AMREX_ALWAYS_ASSERT(names.insert(field.name).second);
        text += field.name + "\n";
    }
    return text;
}

inline void write_manifest (const std::string& path,
                            const GroundMassAccumulationFields& fields)
{
    const auto text = manifest(fields);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream output(path);
        if (!output) amrex::Abort("Cannot create ground mass manifest '" + path + "'");
        output << text;
        output.close();
        if (!output) amrex::Abort("Cannot write ground mass manifest '" + path + "'");
    }
}

inline void validate_manifest (const std::string& path,
                               const GroundMassAccumulationFields& fields,
                               bool verified_legacy_empty = false)
{
    if (!amrex::FileExists(path)) {
        if (fields.empty() && verified_legacy_empty) return;
        amrex::Abort("SuperDroplets checkpoint is missing ground mass manifest '" + path +
                     "'. A legacy empty ground configuration requires matching particle metadata; "
                     "nonempty cumulative histories cannot be recovered from the particle state.");
    }
    amrex::Vector<char> data;
    amrex::ParallelDescriptor::ReadAndBcastFile(path, data);
    if (std::string(data.dataPtr(), data.size()-1) != manifest(fields)) {
        amrex::Abort("SuperDroplets ground mass manifest does not match the configured species "
                     "and aerosols: '" + path + "'");
    }
}

} // namespace erf_ground_mass
#endif
