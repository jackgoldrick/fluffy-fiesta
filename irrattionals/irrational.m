function[x, val, diff] = irrational(prime)
    step = .00000001;
    
    N = 2:step:10;
    j = length(N);

    irs =(prime).^(1./N);
    
  

    for n = 1:j
       c =  abs(irs(n)- pi);
        if c < 0.000000000000000000000000000001 
            
            x = n;

            val = irs(n);
            
            diff = c;
  
            return;

        else 

            x = 0;

            val = 0;

            diff = 0;


        end
   
    end 




end 