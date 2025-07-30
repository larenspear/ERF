#ifndef ERF_PROB_H_
#define ERF_PROB_H_

#include <string>

#include "AMReX_REAL.H"

#include "ERF_ProbCommon.H"

struct ProbParm : ProbParmDefaults {
  amrex::Real rho_0 = 0.0;
  amrex::Real T_0   = 0.0;
  amrex::Real A_0   = 1.0;
  amrex::Real KE_0  = 0.1;  // initialize to calculated rho_hse times input KE
  amrex::Real rhoKE_0 = -1; // alternatively, initialize to specified (rho*KE)

  amrex::Real KE_decay_height = -1;
  amrex::Real KE_decay_order  =  1;

  amrex::Real U_0 = 0.0;
  amrex::Real V_0 = 0.0;
  amrex::Real W_0 = 0.0;

  // random initial perturbations (legacy code)
  amrex::Real U_0_Pert_Mag = 0.0;
  amrex::Real V_0_Pert_Mag = 0.0;
  amrex::Real W_0_Pert_Mag = 0.0;
  amrex::Real T_0_Pert_Mag = 0.0; // perturbation to rho*Theta
  bool pert_rhotheta = true;

  // divergence-free initial perturbations
  amrex::Real pert_deltaU = 0.0;
  amrex::Real pert_deltaV = 0.0;
  amrex::Real pert_periods_U = 5.0;
  amrex::Real pert_periods_V = 5.0;
  amrex::Real pert_ref_height = 100.0;

  // helper vars
  amrex::Real aval;
  amrex::Real bval;
  amrex::Real ufac;
  amrex::Real vfac;

  //RCEMIP
	// Domain configuration
  int use_rcemip = 1;
  int rcemip_case_type = 0;        // 0=small domain, 1=large domain
  int rcemip_rad_scheme = 0;      // 0=default, 1=RCEMIP radiation scheme
  int rcemip_qv_init = 0;         // 0=use default, 1=use RCEMIP specific humidity profile

  amrex::Real rcemip_sst = 300.0;  // SST in K (295, 300 [default], or 305)
  amrex::Real rcemip_domain_height = 40000.0; // Domain height in meters
  // Radiation parameters
  amrex::Real rcemip_co2 = 348.0;            // CO2 concentration in ppmv
  amrex::Real rcemip_ch4 = 1650.0;           // CH4 concentration in ppbv
  amrex::Real rcemip_n2o = 306.0;            // N2O concentration in ppbv

  amrex::Real rcemip_solar_constant = 551.58; // Solar constant in W/m^2
  amrex::Real rcemip_zenith_angle = 42.05;   // Zenith angle in degrees
  amrex::Real rcemip_surface_albedo = 0.07;  // Surface albedo

  // Ozone profile parameters
  amrex::Real rcemip_o3_g1 = 3.6478;         // O3 fit parameter in ppmv hPa^(-g2)
  amrex::Real rcemip_o3_g2 = 0.83209;        // O3 fit parameter
  amrex::Real rcemip_o3_g3 = 11.3515;        // O3 fit parameter in hPa

  // Analytic sounding parameters
  amrex::Real rcemip_tropopause_height = 15000.0; // Tropopause height in meters
  amrex::Real& z_t = rcemip_tropopause_height; // Alias to tropopause
  amrex::Real rcemip_temp_gradient = 0.0067;     // Temperature gradient in K/m
  amrex::Real& rcemip_gamma = rcemip_temp_gradient; // Alias to temperature gradient
  amrex::Real rcemip_surface_pressure = 1014.8;  // Surface pressure in hPa
  amrex::Real rcemip_zq1 = 4000.0;              // Height parameter 1 for humidity in meters
  amrex::Real rcemip_zq2 = 7500.0;              // Height parameter 2 for humidity in meters
  amrex::Real rcemip_qt = 1.0e-11;              // Upper atmosphere specific humidity in g/kg

  // Derived parameters - will be set based on SST
  amrex::Real q0 = 18.65;  // Surface specific humidity in g/kg


}; // namespace ProbParm

class Problem : public ProblemBase
{
public:
    Problem(const amrex::Real* problo, const amrex::Real* probhi);

#include "Prob/ERF_InitConstantDensityHSE.H"
#include "Prob/ERF_InitRayleighDamping.H"

    void init_custom_pert (
        const amrex::Box&  bx,
        const amrex::Box& xbx,
        const amrex::Box& ybx,
        const amrex::Box& zbx,
        amrex::Array4<amrex::Real const> const& state,
        amrex::Array4<amrex::Real      > const& state_pert,
        amrex::Array4<amrex::Real      > const& x_vel_pert,
        amrex::Array4<amrex::Real      > const& y_vel_pert,
        amrex::Array4<amrex::Real      > const& z_vel_pert,
        amrex::Array4<amrex::Real      > const& r_hse,
        amrex::Array4<amrex::Real      > const& p_hse,
        amrex::Array4<amrex::Real const> const& z_nd,
        amrex::Array4<amrex::Real const> const& z_cc,
        amrex::GeometryData const& geomdata,
        amrex::Array4<amrex::Real const> const& mf_m,
        amrex::Array4<amrex::Real const> const& mf_u,
        amrex::Array4<amrex::Real const> const& mf_v,
        const SolverChoice& sc) override;

		void initialize_rcemip_moisture(
    		const amrex::Box& bx,
    		amrex::Array4<amrex::Real> const& state_pert,
    		amrex::Array4<amrex::Real const> const& z_cc,
    		amrex::GeometryData const& geomdata);

    void initialize_rcemip_temp_pressure(
        const amrex::Box& bx,
        amrex::Array4<amrex::Real> const& state_pert,
        amrex::Array4<amrex::Real const> const& z_cc,
        amrex::GeometryData const& geomdata);

protected:
    std::string name() override { return "RCEMIP"; }

private:
    ProbParm parms;
};

#endif
