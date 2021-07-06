function [G] = constraint(X, n_x, N, F)
    h = X(end);
    X = X(1:size(X,1)-1, 1);
    
    % Assuming X is a long vector, get the new indices.
    x_current = X(1:n_x, 1);
    n1 = n_x + 1;
    n2 = 2*n_x;
    x_next = X(n1:n2, 1);
    
    G = x_next - F([x_current; h]);
    
    for n=2:N
        x_current = x_next;
        
        % Assuming X is a long vector, get the new indices.
        n1 = n*n_x + 1;
        n2 = (n+1)*n_x;
        x_next = X(n1:n2, 1);
        
        % Computing the resisduals.
        G = [G; x_next - F([x_current; h])];
    end
end