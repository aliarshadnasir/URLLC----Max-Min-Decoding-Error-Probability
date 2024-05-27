
clc; clear;
% [length(find(a(:,1)<c)) length(find(a(:,3)<c)) length(find(a(:,5)<c)) length(find(a(:,7)<c)) length(find(a(:,9)<c)) length(find(a(:,11)<c))]
% [length(find(a(:,2)<c)) length(find(a(:,4)<c)) length(find(a(:,6)<c)) length(find(a(:,8)<c)) length(find(a(:,10)<c)) length(find(a(:,12)<c))]
only_LSF = 1;
dispi = 0;
V_app = 1;
BW = 5e6; %
Ts = 1/BW;
% T = 100e-6; % frame time
% M = T/Ts;
if (only_LSF == 1)
%     Fig. 1
%     M_vec = [340:20:400];
%     N = 8; %  (number of MTUs)
%     P_tot = 10^(18/10)*1e-3;  
%     
%     Fig. 2
    M_vec = [340:20:400];
    N = 7;
    P_tot = 10^(17/10)*1e-3; 
else % SSF
    %     M_vec = [300:20:400];
    M_vec = [400];
    N = 6;
    P_tot = 10^(25/10)*1e-3;
end
D = 100;
% P_tot = P_tot_tild/Ts;


dmin = 50; dmax = 200;
d_vec = [dmin:(dmax-dmin)/N:dmax-1]';

% d_vec = [100 150 200 250]';
% d_vec = [100:65:300]';


% d_vec = [380 330 280 280 220 240 260 280 220 240]'; % N=2
% d_vec = [280 260 240 220 220 240 260 280 220 240]'; % N=3
d = d_vec(1:N);
g_tol = 1e-3;

beta = 35.3 + 37.6*log10(d);


noise_power_density = 10^((-173)/10)*1e-3; %
s2 = noise_power_density*BW;
if (only_LSF == 1)
    simulations = 1;
else
    simulations = 1000; % Edit this for # of simulations
end

if (simulations == 100)
    load chan_URLLC_mult_ant.mat;
elseif (simulations > 100)
    load chan_hb_thousand.mat;
end
    

epi_vec = zeros(1,length(M_vec));
for M = M_vec
    if (only_LSF ~= 1)
        display([num2str('block-length = ') num2str(M)]);
    end
    epi_s = 0;
    iter_s = 0;
    sim = 1;
    cases = 0; cases_2 = 0;
    while (sim <= simulations)
        if (only_LSF == 1)
            hb = ones(N,1);
        else
            if (simulations > 100)
                hb = hb_thousand(1:N,sim); % 1/sqrt(2)*(randn(N,1)+1j*randn(N,1));
            else
                hb = chan(1,1:N,sim).';
            end
        end

        ht = hb.*sqrt(10.^(-beta/10));
        h = abs(ht).^2/s2;
        
        %%% Assuming equal time allocation
        meq = M/N;
        m = ones(N,1)*meq;
%         peq = P_tot/(N*meq);
        peq = P_tot;
        %%%%
        
        %% Assuming equal power allocation
        
        p = ones(N,1)*peq;
        gamma = p.*abs(ht).^2/s2;
        V = 1 - (1+gamma).^(-2);
        if ( min(log(2)*sqrt(m./V).*(log2(1+gamma)-D./m)) > 0 )
            e_peq = qfunc(log(2)*sqrt(m./V).*(log2(1+gamma)-D./m));
        else
            e_peq = nan;
        end
        
        
        %% only power optimization
        
        cvx_begin quiet
        variables p(N,1) obj_fun;
        expressions gamma(N,1) f(N,1) pow_cons
        
        pow_cons = 0;
        for n = 1:N
            pow_cons = pow_cons + m(n)*p(n)/M;
        end
        gamma = p.*abs(ht).^2/s2;
        if (V_app == 1)
            V = 1;
        else
            V = 1 - (1+gamma).^(-2);
        end
        f = (log(2)*sqrt(m./V).*(log(1+gamma)/log(2) -D./m));
        
        maximize (obj_fun)
        
        subject to
        obj_fun >= 0;
        pow_cons <= P_tot;
        for n = 1:N
            p(n) >= 0;
            f(n) >= obj_fun;
        end
        cvx_end
        
        gamma = p.*abs(ht).^2/s2;
        V = 1 - (1+gamma).^(-2);
        e_popt = qfunc(log(2)*sqrt(m./V).*(log2(1+gamma)-D./m));
        
        
        %% joint power and m optimization
        
        m = ones(N,1)*sqrt(meq);
        p = 1./peq*ones(N,1);
        
        loop_exit = 0;
        iter = 0;
        obj_old = 1000;
        while (loop_exit == 0)
            iter = iter+1;
            mk = m;
            pk = p;
            cvx_begin quiet
            variables p(N,1) m(N,1) obj_fun inv_m(N,1)
            variables q(N,1)
            expressions f(N,1) latenc_const
            
            
            latenc_const = 0;
            for n = 1:N
                latenc_const = latenc_const + m(n)^2;
            end
            %     if (V_app == 1)
            %         V = 1;
            %     else
            %         V = 1 - (1+gamma).^(-2);
            %     end
            f = 2*mk.*log(1+h./pk) + ((mk.*h)./(h+pk)).*(1-p./pk) - log(1+h./pk).*(mk.^2).*inv_m  - D*log(2)*inv_m;
            
            maximize (obj_fun)
            
            subject to
            
%             obj_fun >= 0;
            sum(q) <= P_tot*M;
            latenc_const <= M;
            for n = 1:N
                m(n) >= 0; p(n) >= 0; inv_m(n) >= 0;
                [inv_m(n) 1 ; 1 m(n)] == semidefinite(2);
                f(n) >= obj_fun;
                [q(n) m(n) ; m(n) p(n)] == semidefinite(2);
            end
            cvx_end
            
            
            ptrue = 1./p;
            mtrue = m.^2;
            
            gamma = ptrue.*abs(ht).^2/s2;
            V = 1 - (1+gamma).^(-2);
            e_mpopt = qfunc(log(2)*sqrt(mtrue./V).*(log2(1+gamma)-D./mtrue));
            
            if (dispi == 1)
                display([num2str(iter) ' f_obj_fun ' num2str(obj_fun) ' f_exp '  num2str(min(log(2)*sqrt(mtrue./V).*(log2(1+gamma)-D./mtrue))) ]);
            end
            
            if ( abs((obj_fun - obj_old)/obj_fun) < g_tol )
                loop_exit = 1;
            end
            obj_old = obj_fun;
            
        end
        
        p = 1./p;
        m = floor(m.^2);
        
        %%% greedy search 
        
        exit_loop = 0;
        while( sum(m) < M && exit_loop == 0 )
            gamma = p.*abs(ht).^2/s2;
            V = 1 - (1+gamma).^(-2);
            e_mpopt = qfunc(log(2)*sqrt(m./V).*(log2(1+gamma)-D./m));
            [maxi indi] = max(e_mpopt);
            if ( m'*p/M > P_tot )
                exit_loop = 1;
            else
                m(indi) = m(indi)+1;
            end
        end
        
        %%%%%%%%%%%%%%%%%
        
        
        gamma = p.*abs(ht).^2/s2;
        V = 1 - (1+gamma).^(-2);
        e_mpopt = qfunc(log(2)*sqrt(m./V).*(log2(1+gamma)-D./m));
        
        %         if (dispi == 1)
        if (only_LSF ~= 1)
            display([num2str(sim) ' ' num2str(max(e_mpopt)) ' ' num2str(max(e_popt))  ' ' num2str(iter) ]);
        end
        %         end
        if (max(e_mpopt) <= 1e-9)
            cases = cases + 1;
        end
        if (max(e_popt) <= 1e-9)
            cases_2 = cases_2 + 1;
        end
        
        epi_s = epi_s + max(e_mpopt);
        iter_s = iter_s + iter;
        sim = sim+1;
    end
    
    if (only_LSF == 1)
        epi_vec(M==M_vec) =  epi_s;
        display([num2str(M) ' Alg1 ' num2str(epi_s) ' eq_BL ' num2str(max(e_popt)) ' eq_BLP ' num2str(max(e_peq)) ' ' num2str(iter_s) ]);
    else
        epi_vec(M==M_vec) =  cases/simulations;
        display([num2str(M) ' ' num2str(cases/simulations) ' ' num2str(cases_2/simulations)  ' ' num2str(iter_s/cases) ]);
    end
    
     
end
semilogy(M_vec,epi_vec);
epi_vec'
