def compute_error(u_h, u_exact, norm_type="L2"):
    # Calculate error norms
    return errornorm(u_exact, u_h, norm_type)