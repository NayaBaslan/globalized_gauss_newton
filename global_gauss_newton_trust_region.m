% Globalization of Gauss Newton with
%   F: R^n -> R^m Objective function
%   G: R^n -> R^l Equality constraint
%   param: Parameters containing
%     delta in (0,1)
%     x_0   in R^n
%     gamma in (0,1)
%     mu_bar vector in (0,infinity)^2 where mu_bar(1) < mu_bar(2).

function [ x_star, x_all, F_star, J ] = global_gauss_newton_trust_region(F, G, param )
    import casadi.*
    
    %% Functions
    psi = @(x,mu)   norm(F(x)) + mu*norm(G(x));

    %% Step (0) (INIT)
    alpha(1) = 1;
    k         = 1;
    mu        = [0 1];
    
    % Parameters
    delta_1   = param.delta_1;
    delta_2   = param.delta_2;
    x(:, 1) = param.x_0;
    gamma_1 = param.gamma_1;
    gamma_2 = param.gamma_2;
    
    gamma_3 = param.gamma_3;
    gamma_4 = param.gamma_4;
    mu_bar = param.mu_bar;
    epsilon = 10^(-8);       % tolerance.
    
    alpha_0 = 1;
    % Casadi variable
    n_x = size(x(:, 1), 1);
    u = SX.sym('u', n_x);
    F_sym = F(u);
    G_sym = G(u);
    F_cas = Function('F',{u},{F_sym, jacobian(F_sym,u)});
    G_cas = Function('G',{u},{G_sym, jacobian(G_sym,u)});
    n_F = size(F_sym, 1);
    n_G = size(G_sym, 1);
    
    %% THE ALGORITHM.
    while(true)
        %% Step (1)
        % Using Casadi to differentiate.
        [ F_k, JF_k_cas ] = F_cas(x(:, k));
        [ G_k, JG_k_cas ] = G_cas(x(:, k));
        
        F_k = full(F_k);
        F_k_prime = full(JF_k_cas);
        G_k = full(G_k);
        G_k_prime = full(JG_k_cas);
        
        %% Step (2)
            G_k_prime_plus = pinv(G_k_prime, 10^(-6));

            % Orthoprojector.
            I_n_x = eye(n_x);
            E = I_n_x - G_k_prime_plus*G_k_prime;

            % Save important values
            G_pp_G = G_k_prime_plus * G_k;
            F_p_G_pp_G = F_k_prime * G_pp_G;
            F_p_E_p = pinv(F_k_prime*E, 10^(-6));
            P = F_k_prime*E*F_p_E_p;

            % Computing d using MPI.
            d_tilde(:, k) = -G_pp_G + F_p_E_p*(F_p_G_pp_G - F_k);
        %end
        %fprintf('d_tilde - d = %f\n', norm(d(:, k) - d_tilde(:, k)));
        if (norm(d_tilde(:, k)) <= epsilon)
            break;
        end
        
        %% Step (3)
        % Finally compute return value.
        I_n_F = eye(n_F);
        denom = (norm(F_k) + norm(F_k_prime*d_tilde(:, k) + F_k))*norm(G_k);
        if ( denom > eps )
            num_1 = F_k + (I_n_F - P)*(F_k - F_p_G_pp_G);
            num_2 = (I_n_F - P)*F_p_G_pp_G;
            
            omega(k) = (num_1'*num_2)/denom;
        else
            omega(k) = 0;
        end
        if ( mu(1) >= abs(omega(k)) + mu_bar(1) ) 
            mu(2) = mu(1);
        else 
            mu(2) = abs(omega(k)) + mu_bar(2);
        end
        
        %% Step (4)
        % Update
        if (k > 1)
            alpha(k) = max(alpha(k-1)*norm(d_tilde(:, k-1))/norm(d_tilde(:, k)),gamma_1*alpha(k-1) );
        end
        
        %% Step (5)
        alpha_n = alpha(k);
        while(true)
            psi_1 = psi(x(:,k), mu(2));
            psi_2n = psi(x(:,k) + alpha_n*d_tilde(:, k), mu(2));
            p_n = alpha_n*d_tilde(:, k);
            phi_n = norm(F_k_prime*p_n+F_k) + mu(2)*norm(G_k_prime*p_n+G_k);
                        
            if (psi_1 - psi_2n < delta_2*(psi_1 - phi_n) )
                break;
            end
            alpha_0 = alpha_n;
            alpha_n = min(gamma_3*alpha_n,1);
        end
        
        if (psi_1 - psi_2n >= delta_1*(psi_1 - phi_n) )
            alpha(k) = alpha_n;
        else
            alpha(k) = alpha_0;
        end
        while(true)
            psi_2 = psi(x(:,k) + alpha(k)*d_tilde(:, k), mu(2));
            p = alpha(k)*d_tilde(:, k);
            phi = norm(F_k_prime*p+F_k) + mu(2)*norm(G_k_prime*p+G_k);
                        
            if (psi_1 - psi_2 >= delta_1*(psi_1 - phi) )
                break;
            end
            alpha(k) = gamma_1*alpha(k);
        end
               
        fprintf('k = %d || alpha(k) = %e || norm(d) = %e || Res = %e\n', k, alpha(k), norm(d_tilde(:, k)), norm(F_k));
        
        %% Step (6)
        x(:, k+1) = x(:, k) + alpha(k)*d_tilde(:, k);
        alpha_old = alpha(k);
        k = k + 1;
        
        if( norm(x(:, k) - x(:, k-1)) <= 10^(-8) )
            break;
        end
        
        %  TAKE OUT EVENTUALLY
        if (k  > 1000)
            break;
        end
    end
    
    %% return val
    x_star = x(:, k);
    x_all  = x;
    F_star = F_k;
    J      = F_k_prime; 
end