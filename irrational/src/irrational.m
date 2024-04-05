function[x, val, diff] = irrational(prime)
    step = .00000001;
    
    N = 9.880414053:step:9.9;
    j = length(N);

    irs =(prime).^(1./N);
    
    err_desired = 0.00000001;
         x = 0;

            val = 0;

            diff = 0;

    for n = 1:j
       c =  abs(irs(n)- pi);
        if c < err_desired
            
            x = N(n);

            val = irs(n);
            
            diff = c;
  
            return;


        end
   
    end 




end 