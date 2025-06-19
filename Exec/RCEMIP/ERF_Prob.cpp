#include "ERF_Prob.H"
#include "AMReX_Random.H"

using namespace amrex;

std::unique_ptr<ProblemBase>
amrex_probinit(const amrex_real* problo, const amrex_real* probhi)
{
    return std::make_unique<Problem>(problo, probhi);
}

Problem::Problem(const amrex::Real* problo, const amrex::Real* probhi)
{
  // Parse params
  ParmParse pp("prob");

  // Add below your existing parameter queries
  pp.query("use_rcemip", parms.use_rcemip); // Use RCEMIP initial conditions
  pp.query("rcemip_case_type", parms.rcemip_case_type); // 0 = small domain, 1 = large domain
  pp.query("rcemip_sst", parms.rcemip_sst);  // Surface temperature for RCEMIP cases
  pp.query("rcemip_domain_height", parms.rcemip_domain_height);  // Domain height for RCE simulations
  pp.query("rcemip_rad_scheme", parms.rcemip_rad_scheme);  // Radiation scheme to use
  pp.query("rcemip_qv_init", parms.rcemip_qv_init);  // Initial water vapor profile

  pp.query("rho_0", parms.rho_0);
  pp.query("T_0", parms.T_0);
  pp.query("A_0", parms.A_0);
  pp.query("KE_0", parms.KE_0);
  pp.query("rhoKE_0", parms.rhoKE_0);
  pp.query("KE_decay_height", parms.KE_decay_height);
  pp.query("KE_decay_order", parms.KE_decay_order);

  pp.query("U_0", parms.U_0);
  pp.query("V_0", parms.V_0);
  pp.query("W_0", parms.W_0);
  pp.query("U_0_Pert_Mag", parms.U_0_Pert_Mag);
  pp.query("V_0_Pert_Mag", parms.V_0_Pert_Mag);
  pp.query("W_0_Pert_Mag", parms.W_0_Pert_Mag);
  pp.query("T_0_Pert_Mag", parms.T_0_Pert_Mag);
  pp.query("pert_rhotheta", parms.pert_rhotheta);

  pp.query("pert_deltaU", parms.pert_deltaU);
  pp.query("pert_deltaV", parms.pert_deltaV);
  pp.query("pert_periods_U", parms.pert_periods_U);
  pp.query("pert_periods_V", parms.pert_periods_V);
  pp.query("pert_ref_height", parms.pert_ref_height);
  parms.aval = parms.pert_periods_U * 2.0 * PI / (probhi[1] - problo[1]);
  parms.bval = parms.pert_periods_V * 2.0 * PI / (probhi[0] - problo[0]);
  parms.ufac = parms.pert_deltaU * std::exp(0.5) / parms.pert_ref_height;
  parms.vfac = parms.pert_deltaV * std::exp(0.5) / parms.pert_ref_height;

  if (parms.rcemip_sst == 295.0) { //Because these are read in from files, float == is ok?
    parms.q0 = parms.rcemip_sst = 12.00
  } else if (parms.rcemp_sst == 300.0){
    parms.q0 = parms.rcemip_sst = 18.65;
  } else if (parms.rcemip_sst == 305.0) {
    parms.q0 = parms.rcemip_sst = 24.00;
  } else {
    AMREX_ASSERT_WITH_MESSAGE(false, "Invalid SST for RCEMIP case"); //Do we only want to allow paper values?
  }

 	if (parms.use_rcemip) {
  		amrex::Print() << "\n==== RCEMIP Configuration ====" << std::endl;
    	amrex::Print() << " - Case type: " << (parms.rcemip_case_type == 0 ? "RCE_small" : "RCE_large") << std::endl;
    	amrex::Print() << " - SST: " << parms.rcemip_sst << " K" << std::endl;
    	amrex::Print() << " - Surface q0: " << parms.rcemip_q0 << " g/kg" << std::endl;
    	amrex::Print() << " - Domain height: " << parms.rcemip_domain_height << " m" << std::endl;
    	amrex::Print() << "==========================\n" << std::endl;
  }

  init_base_parms(parms.rho_0, parms.T_0);
}

void
Problem::init_custom_pert(
    const amrex::Box&  bx,
    const amrex::Box& xbx,
    const amrex::Box& ybx,
    const amrex::Box& zbx,
    amrex::Array4<amrex::Real const> const& /*state*/,
    amrex::Array4<amrex::Real      > const& state_pert,
    amrex::Array4<amrex::Real      > const& x_vel_pert,
    amrex::Array4<amrex::Real      > const& y_vel_pert,
    amrex::Array4<amrex::Real      > const& z_vel_pert,
    amrex::Array4<amrex::Real      > const& r_hse,
    amrex::Array4<amrex::Real      > const& /*p_hse*/,
    amrex::Array4<amrex::Real const> const& z_nd,
    amrex::Array4<amrex::Real const> const& z_cc,
    amrex::GeometryData const& geomdata,
    amrex::Array4<amrex::Real const> const& /*mf_m*/,
    amrex::Array4<amrex::Real const> const& /*mf_u*/,
    amrex::Array4<amrex::Real const> const& /*mf_v*/,
    const SolverChoice& sc)
{
    const bool use_moisture = (sc.moisture_type != MoistureType::None);

    if (parms.KE_decay_height > 0) {
        amrex::Print() << "Initial KE profile (order " << parms.KE_decay_order
                       << ") will extend up to " << parms.KE_decay_height
                       << std::endl;
    }

    if (parms.pert_ref_height > 0) {
        if ((parms.pert_deltaU != 0.0) || (parms.pert_deltaV != 0.0)) {
            amrex::Print() << "Adding divergence-free perturbations "
                           << parms.pert_deltaU << " " << parms.pert_deltaV
                           << std::endl;
        }
        if (parms.U_0_Pert_Mag != 0.0) {
            amrex::Print() << "Adding random x-velocity perturbations" << std::endl;
        }
        if (parms.V_0_Pert_Mag != 0.0) {
            amrex::Print() << "Adding random y-velocity perturbations" << std::endl;
        }
        if (parms.T_0_Pert_Mag != 0.0) {
            if (parms.pert_rhotheta) {
                amrex::Print() << "Adding random rho*theta perturbations" << std::endl;
            } else {
                amrex::Print() << "Adding random theta perturbations" << std::endl;
            }
        }
    }

  ParallelForRNG(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
    // Geometry
    const Real* prob_lo = geomdata.ProbLo();
    const Real* prob_hi = geomdata.ProbHi();
    const Real* dx = geomdata.CellSize();
    const Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const Real z = (z_cc) ? z_cc(i,j,k) : prob_lo[2] + (k + 0.5) * dx[2];

    // Define a point (xc,yc,zc) at the center of the domain
    const Real xc = 0.5 * (prob_lo[0] + prob_hi[0]);
    const Real yc = 0.5 * (prob_lo[1] + prob_hi[1]);
    const Real zc = 0.5 * (prob_lo[2] + prob_hi[2]);

    const Real r  = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));

    // Add temperature perturbations
    if ((z <= parms_d.pert_ref_height) && (parms_d.T_0_Pert_Mag != 0.0)) {
        Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
        state_pert(i, j, k, RhoTheta_comp) = (rand_double*2.0 - 1.0)*parms_d.T_0_Pert_Mag;
        if (!parms_d.pert_rhotheta) {
            // we're perturbing theta, not rho*theta
            state_pert(i, j, k, RhoTheta_comp) *= r_hse(i,j,k);
        }
    }

    // Set scalar = A_0*exp(-10r^2), where r is distance from center of domain
    state_pert(i, j, k, RhoScalar_comp) = parms_d.A_0 * exp(-10.*r*r);

    // Set an initial value for SGS KE
    if (state_pert.nComp() > RhoKE_comp) {
        // Deardorff
        if (parms_d.rhoKE_0 > 0) {
            state_pert(i, j, k, RhoKE_comp) = parms_d.rhoKE_0;
        } else {
            state_pert(i, j, k, RhoKE_comp) = r_hse(i,j,k) * parms_d.KE_0;
        }
        if (parms_d.KE_decay_height > 0) {
            // scale initial SGS kinetic energy with height
            state_pert(i, j, k, RhoKE_comp) *= max(
                std::pow(1 - min(z/parms_d.KE_decay_height,1.0), parms_d.KE_decay_order),
                1e-12);
        }
    }

    if (use_moisture) {
        state_pert(i, j, k, RhoQ1_comp) = 0.0;
        state_pert(i, j, k, RhoQ2_comp) = 0.0;
    }
  });

  // Set the x-velocity
  ParallelForRNG(xbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();
    const Real y = prob_lo[1] + (j + 0.5) * dx[1];
    const Real z = (z_nd) ? 0.25*( z_nd(i,j  ,k) + z_nd(i,j  ,k+1)
                                 + z_nd(i,j+1,k) + z_nd(i,j+1,k+1) )
                               : prob_lo[2] + (k + 0.5) * dx[2];

    // Set the x-velocity
    x_vel_pert(i, j, k) = parms_d.U_0;
    if ((z <= parms_d.pert_ref_height) && (parms_d.U_0_Pert_Mag != 0.0))
    {
        Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
        Real x_vel_prime = (rand_double*2.0 - 1.0)*parms_d.U_0_Pert_Mag;
        x_vel_pert(i, j, k) += x_vel_prime;
    }
    if (parms_d.pert_deltaU != 0.0)
    {
        const amrex::Real yl = y - prob_lo[1];
        const amrex::Real zl = z / parms_d.pert_ref_height;
        const amrex::Real damp = std::exp(-0.5 * zl * zl);
        x_vel_pert(i, j, k) += parms_d.ufac * damp * z * std::cos(parms_d.aval * yl);
    }
  });

  // Set the y-velocity
  ParallelForRNG(ybx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx = geomdata.CellSize();
    const Real x = prob_lo[0] + (i + 0.5) * dx[0];
    const Real z = (z_nd) ? 0.25*( z_nd(i  ,j,k) + z_nd(i  ,j,k+1)
                                 + z_nd(i+1,j,k) + z_nd(i+1,j,k+1) )
                               : prob_lo[2] + (k + 0.5) * dx[2];

    // Set the y-velocity
    y_vel_pert(i, j, k) = parms_d.V_0;
    if ((z <= parms_d.pert_ref_height) && (parms_d.V_0_Pert_Mag != 0.0))
    {
        Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
        Real y_vel_prime = (rand_double*2.0 - 1.0)*parms_d.V_0_Pert_Mag;
        y_vel_pert(i, j, k) += y_vel_prime;
    }
    if (parms_d.pert_deltaV != 0.0)
    {
        const amrex::Real xl = x - prob_lo[0];
        const amrex::Real zl = z / parms_d.pert_ref_height;
        const amrex::Real damp = std::exp(-0.5 * zl * zl);
        y_vel_pert(i, j, k) += parms_d.vfac * damp * z * std::cos(parms_d.bval * xl);
    }
  });

  // Set the z-velocity
  ParallelForRNG(zbx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k, const amrex::RandomEngine& engine) noexcept {
    const int dom_lo_z = geomdata.Domain().smallEnd()[2];
    const int dom_hi_z = geomdata.Domain().bigEnd()[2];

    // Set the z-velocity
    if (k == dom_lo_z || k == dom_hi_z+1)
    {
        z_vel_pert(i, j, k) = 0.0;
    }
    else if (parms_d.W_0_Pert_Mag != 0.0)
    {
        Real rand_double = amrex::Random(engine); // Between 0.0 and 1.0
        Real z_vel_prime = (rand_double*2.0 - 1.0)*parms_d.W_0_Pert_Mag;
        z_vel_pert(i, j, k) = parms_d.W_0 + z_vel_prime;
    }
  });
}

//=============================================================================
// USER-DEFINED FUNCTION
//=============================================================================
//
void Problem::initialize_rcemip_moisture(
    const amrex::Box& bx,
    amrex::Array4<amrex::Real> const& state_pert,
    amrex::Array4<amrex::Real const> const& z_cc,
    amrex::GeometryData const& geomdata)
{
    const Real z_q1 = 4000.0;
    const Real z_q2 = 7500.0;
    const Real z_t = 15000.0;
    const Real q_t = 1.0e-11;
    const Real q0 = parms.rcemip_q0

    ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      const Real* prob_lo = geomdata.ProbLo();
      const Real* dx = geomdata.CellSize();
      const Real z = (z_cc) ? z_cc(i,j,k) : prob_lo[2] + (k + 0.5) * dx[2];
      Real qv = 0.0;
      if (z <= z_t){
          qv = qt;
      } else {
        const Real ratio = z/z_q2;
        qv = q0 * std::exp(-z/z_q1) * std::exp(-ratio * ratio);
      }
    });

    state_pert(i, j, k, RhoQ1_comp) = qv;
}
//=============================================================================
// USER-DEFINED FUNCTION
//=============================================================================
//
void Problem::initialize_rcemip_temp(
    const amrex::Box& bx,
    amrex::Array4<amrex::Real> const& state_pert,
    amrex::Array4<amrex::Real const> const& z_cc,
    amrex::GeometryData const& geomdata)
{
    const Real gamma = 0.0067;  // Dry adiabatic lapse rate
    const Real T_0 = parms.rcemip_sst;  // Surface temperature equals SST
    const Real z_t = 15000.0;  // Tropopause height (m)
    const Real q0 = parms.rcemip_q0;  // Surface specific humidity

    const p0 = 1014.8; // Surface pressure (hPa)
    const Rd = 287.04; // Dry air gas constant (J/(kg*K))
    const g = 9.79764; // Gravitational acceleration (m/s^2)

    ParallelFor(bx, [=, parms_d=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const Real* prob_lo = geomdata.ProbLo();
        const Real* dx = geomdata.CellSize();
        const Real z = (z_cc) ? z_cc(i,j,k) : prob_lo[2] + (k + 0.5) * dx[2];

        Real T_v0 = T_0 * (1 + 0.608 * q0);
        Real T_vt = T_v0 - gamma * z_t;

        if (z <= z_tropo) {
            // Below tropopause: linear decrease with height
            T_v = T_v0 - gamma * z;
        } else {
            // Above tropopause: isothermal
            T_v = T_vt;
        }

        Real T = T_v / (1 + 0.608 * qv);

        p_t = p0 * std::pow((T_vt/T_v0), (g/(Rd*gamma)));

        if (z <= z_tropo) {
            // Below tropopause: hydrostatic balance
            p = p0 * std::pow(T_v0 - (gamma * z) / T_v0, (g/(Rd*gamma)));
        } else {
            // Above tropopause: isothermal hydrostatic balance
            p = p_t * std::exp(-g*(z-z_t)/(Rd*T_vt));
        }

    });

    state_pert(i, j, k, RhoTheta_comp) = p / (Rd * T);
    state_pert(i, j, k, RhoQ1_comp) = qv;
}
