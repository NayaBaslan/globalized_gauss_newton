% Globalization of Gauss Newton with
%   F: R^n -> R^m Objective function
%   G: R^n -> R^l Equality constraint
%   param: Parameters containing
%     delta in (0,1)
%     x_0   in R^n
%     gamma in (0,1)
%     mu_bar vector in (0,infinity)^2 where mu_bar(1) < mu_bar(2).

function [ x_star, F_k_prime ] = global_gauss_newton_fast(F, G, param )
    import casadi.*
    
    %% Functions
    psi = @(x,mu)   norm(F(x)) + mu*norm(G(x));

    %% Step (0) (INIT)
    alpha_old  = 1;
    k          = 1;
    mu         = [0 1];
    
    % Parameters
    delta   = param.delta;
    x_star  = param.x_0;
    gamma   = param.gamma;
    mu_bar  = param.mu_bar;
    epsilon = 10^(-16);       % tolerance.
    
    % Casadi variable
    n_x = size(x_star, 1);
    u = SX.sym('u', n_x);
    F_sym = F(u);
    G_sym = G(u);
    F_cas = Function('F',{u},{F_sym, jacobian(F_sym,u)});
    G_cas = Function('G',{u},{G_sym, jacobian(G_sym,u)});
    n_F = size(F_sym, 1);
    
    %% THE ALGORITHM.
    while(true)
        %% Step (1)
        % Using Casadi to differentiate.
        [ F_k, JF_k_cas ] = F_cas(x_star);
        [ G_k, JG_k_cas ] = G_cas(x_star);
        
        F_k = full(F_k);
        F_k_prime = full(JF_k_cas);
        G_k = full(G_k);
        G_k_prime = full(JG_k_cas);
        
        %% Step (2)
        G_k_prime_plus = pinv(G_k_prime, 10^(-9));

        % Orthoprojector.
        I_n_x = eye(n_x);
        E = I_n_x - G_k_prime_plus*G_k_prime;

        % Save important values
        G_pp_G = G_k_prime_plus * G_k;
        F_p_G_pp_G = F_k_prime * G_pp_G;
        F_p_E_p = pinv(F_k_prime*E, 10^(-6));
        P = F_k_prime*E*F_p_E_p;

        % Computing d using MPI.
        d_tilde = -G_pp_G + F_p_E_p*(F_p_G_pp_G - F_k);
    
        if (norm(d_tilde) <= epsilon)
            break;
        end
        
        %% Step (3)
        % fprintf('Step (3) || Updating penalty.\n');

        % Finally compute return value.
        I_n_F = eye(n_F);
        denom = (norm(F_k) + norm(F_k_prime*d_tilde + F_k))*norm(G_k);
        if ( denom > eps )
            num_1 = F_k + (I_n_F - P)*(F_k - F_p_G_pp_G);
            num_2 = (I_n_F - P)*F_p_G_pp_G;
            
            omega = (num_1'*num_2)/denom;
        else
            omega = 0;
        end
        if ( mu(1) >= abs(omega) + mu_bar(1) ) 
            mu(2) = mu(1);
        else 
            mu(2) = abs(omega) + mu_bar(2);
        end
        
        %% Step (4)
        % Update
        alpha = min(alpha_old/gamma, 1);
        
        
        %% Step (5)
        while(true)
            psi_1 = psi(x_star, mu(2));
            psi_2 = psi(x_star + alpha*d_tilde, mu(2));
            p = alpha*d_tilde;
            phi = norm(F_k_prime*p+F_k) + mu(2)*norm(G_k_prime*p+G_k);
                        
            if (psi_1 - psi_2 >= delta*(psi_1 - phi) )
                break;
            end
            alpha = gamma*alpha;
        end
        
        %% Step (6)
        x_star = x_star + alpha*d_tilde;
        k = k + 1;
                
        %  TAKE OUT EVENTUALLY
        if (k  > 200)
            break;
        end
    end
end