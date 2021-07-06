% Globalization of Gauss Newton with
%   F: R^n -> R^m Objective function
%   G: R^n -> R^l Equality constraint
%   param: Parameters containing
%     delta in (0,1)
%     x_0   in R^n
%     gamma in (0,1)
%     mu_bar vector in (0,infinity)^2 where mu_bar(1) < mu_bar(2).

function [ x_star, x_all, F_star, J, F_norm_all, G_norm_all , alpha ] = global_gauss_newton(F, G, param )
    import casadi.*
    
    %% Functions
    psi = @(x,mu)   norm(F(x)) + mu*norm(G(x));

    %% Step (0) (INIT)
    alpha(1)  = 1;
    k         = 1;
    mu        = [0 1];
    
    % Parameters
    delta = param.delta;
    x(:, 1) = param.x_0;
    gamma = param.gamma;
    mu_bar = param.mu_bar;
    epsilon = 10^(-16);       % tolerance.
    
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
        
        % COMPUTING JACOBIANS
        % Step to take on each dimension (has to be small enough for precision)
        %h = sqrt(eps);
        
        %% Step (2)
        %warning('off', 'all')
        % fprintf('Step (2) || Computing d.\n')
        % Testing against full rank assumptions
        %if (rank(G_k_prime) < n_G || rank([F_k_prime; G_k_prime]) < n_x)
            %d_tilde(:, k) = lsqlin(F_k_prime, -F_k, zeros(1, n_x), 0, G_k_prime, -G_k);
        %warning('on', 'all')
        
            %else
            % Computing important values
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
            d_tilde(:, k) = -G_pp_G + F_p_E_p*(F_p_G_pp_G - F_k);
        %end
        %fprintf('d_tilde - d = %f\n', norm(d(:, k) - d_tilde(:, k)));
        if (norm(d_tilde(:, k)) <= epsilon)
            break;
        end
        
        %% Step (3)
        % fprintf('Step (3) || Updating penalty.\n');

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
            fprintf('mu(1) = %f\n',mu(1));
            fprintf('mu(2) = %f\n',mu(2));
        else 
            mu(2) = abs(omega(k)) + mu_bar(2);
        end
        
        %% Step (4)
        % Update
        if (k>1)
            alpha(k) = min(alpha(k-1)/gamma, 1);
        end
        
        %% Step (5)
        while(true)
            psi_1 = psi(x(:,k), mu(2));
            psi_2 = psi(x(:,k) + alpha(k)*d_tilde(:, k), mu(2));
            p = alpha(k)*d_tilde(:, k);
            phi = norm(F_k_prime*p+F_k) + mu(2)*norm(G_k_prime*p+G_k);
                        
            if (psi_1 - psi_2 >= delta*(psi_1 - phi) )
                break;
            end
            alpha(k) = gamma*alpha(k);
            
        end
        
        F_norm_all(k, 1) = norm(F_k);
        G_norm_all(k, 1) = norm(G_k);
        fprintf('k = %d || alpha(k) = %e || norm(d) = %e || norm(F) = %e || norm(G) = %e\n', k, alpha(k), norm(d_tilde(:, k)), F_norm_all(k, 1), G_norm_all(k,1));
        
        %% Step (6)
        x(:, k+1) = x(:, k) + alpha(k)*d_tilde(:, k);
        k = k + 1;
        
        if( norm(x(:, k) - x(:, k-1)) <= 10^(-12) )
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