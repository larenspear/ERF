    // RCEMIP (Wing et al. 2018, GMD 11, 793-813): the initial state comes from
    // the analytic sounding (see Exec/CanonicalTests/RCEMIP); here we only add
    // the protocol's symmetry-breaking thermal noise (Sec. 3.2.3): random
    // perturbations in the five lowest model layers, with an amplitude of
    // 0.1 K in the lowest layer decreasing linearly to 0.02 K in the fifth.
    ParmParse pp("prob");

    Real T_pert_amp_bottom = amrex::Real(0.10); pp.query("T_pert_amp_bottom", T_pert_amp_bottom);
    Real T_pert_amp_top    = amrex::Real(0.02); pp.query("T_pert_amp_top"   , T_pert_amp_top);
    int  T_pert_num_layers = 5;                 pp.query("T_pert_num_layers", T_pert_num_layers);
    Real qv_0_Pert_Mag     = zero;              pp.query("qv_0_Pert_Mag"    , qv_0_Pert_Mag);

    const bool use_moisture = (sc.moisture_type != MoistureType::None);

    const int klo = geomdata.Domain().smallEnd(2);

    ParallelForRNG(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k, const RandomEngine& engine) noexcept
    {
        const int layer = k - klo;

        //
        // Add potential-temperature perturbations. We perturb (rho theta)
        // rather than using the constant-pressure density trick because the
        // anelastic solver holds density to the base state and would discard
        // a pure density perturbation, leaving the flow unseeded.
        //
        if ((layer >= 0) && (layer < T_pert_num_layers) && (T_pert_amp_bottom != zero)) {

            // Amplitude decreases linearly from the bottom layer to the top
            // perturbed layer (0.1 K -> 0.02 K over five layers in the protocol)
            Real frac = (T_pert_num_layers > 1) ? Real(layer) / Real(T_pert_num_layers-1) : zero;
            Real amp  = T_pert_amp_bottom + (T_pert_amp_top - T_pert_amp_bottom) * frac;

            Real rho = state(i,j,k,Rho_comp);

            Real rand_double = amrex::Random(engine); // Between zero and one
            state_pert(i, j, k, RhoTheta_comp) = rho * (rand_double*two - one)*amp;
        }

        // Optional moisture noise (off by default; the protocol asks for
        // thermal noise only)
        if (use_moisture && (layer >= 0) && (layer < T_pert_num_layers) &&
            (qv_0_Pert_Mag != zero))
        {
            Real rhoold = state(i,j,k,Rho_comp);
            Real rhonew = rhoold + state_pert(i,j,k,Rho_comp);

            Real qvold = state(i,j,k,RhoQ1_comp) / rhoold;

            Real rand_double = amrex::Random(engine); // Between zero and one
            Real qvnew = qvold + (rand_double*two - one)*qv_0_Pert_Mag;

            state_pert(i, j, k, RhoQ1_comp) = rhonew * qvnew - rhoold * qvold;
        }
    });
