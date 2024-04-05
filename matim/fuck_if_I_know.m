% Name: Jack Goldrick
% Date: 7/2/23

%% Housekeeping

clc
clear
close all;

%% Import Images datasets

X = imread('square.jpg');
Y = imread('squareedita.jpg');

%figure(3)
%mshow(X);
%figure(4)
%imshow(Y);

    %% Separate RGB
    XR = X(:,:,1);
    XG = X(:,:,2);
    XB = X(:,:,3);

    YR = Y(:,:,1);
    YG = Y(:,:,2);
    YB = Y(:,:,3);

    %% Create Greyscale
    Xg = rgb2gray(X);
    Yg = rgb2gray(Y);
    
    %% Normalize
    Xgray = double(Xg)/255;
    Ygray = double(Yg)/255;

    XR = double(XR)/255;
    XG = double(XG)/255;
    XB = double(XB)/255;

    YR = double(YR)/255;
    YG = double(YG)/255;
    YB = double(YB)/255;

    %% Size Check
    [Xm,Xn] = size(Xgray);
    [Ym,Yn] = size(Ygray);


    Xg_n120 = Xgray-120;
    Yg_n120 = Ygray-120;

%{
figure(5)
imagesc(Xgray);
figure(6)
imagesc(Ygray);


%figure(3)
%imshow(Xgray);
%figure(4)
%imshow(Ygray);

%}

if (Xm ~= Xn && Ym ~= Yn) 
    
    Xds = transpose(Xgray)*(Xgray);
    Yds = transpose(Ygray)*(Ygray);

    [Xds_v,Xds_e] = eig(X_ds,"matrix");
    [Yds_v,Yds_e] = eig(Y_ds,"matrix");
    
else 
    %% Eigen Decomp of Grey Image and Channels   
    [Xgray_v,Xgray_e] = eig(Xgray,"matrix");
    [Ygray_v,Ygray_e] = eig(Ygray,"matrix");

    [XR_v,XR_e] = eig(XR,"vector");
    [XG_v,XG_e] = eig(XG,"vector");
    [XB_v,XB_e] = eig(XB,"vector");

    [YR_v,YR_e] = eig(YR,"vector");
    [YG_v,YG_e] = eig(YG,"vector");
    [YB_v,YB_e] = eig(YB,"vector");


    %% Pixel Variance
    CompXY = abs(Xgray - Ygray);
    CompXYR = abs(XR - YR);
    CompXYG = abs(XG - YG);
    CompXYB = abs(XB - YB);



    %% gram matrix
    Xgram = transpose(Xgray)*(Xgray);
    XgramR = transpose(XR) * XR;
    XgramG = transpose(XG) * XG;
    XgramB = transpose(XB) * XB;

    Ygram = transpose(Ygray)*(Ygray);
    YgramR = transpose(YR) * YR;
    YgramG = transpose(YG) * YG;
    YgramB = transpose(YB) * YB;

    CompGram = transpose(CompXY) * CompXY;
    CompGramR = transpose(CompXYR) * CompXYR;
    CompGramG = transpose(CompXYG) * CompXYG;
    CompGramB = transpose(CompXYB) * CompXYB;
    
    %% inner product comparison

    XYin = transpose(Xgray)*(Ygray);


    %% Eigen the shit out of gram

    [XYin_v,XYin_e] = eig(XYin,"matrix");

    [Xgram_v,Xgram_e] = eig(Xgram,"matrix");
    [XgramR_v,XgramR_e] = eig(XgramR,"vector");
    [XgramG_v,XgramG_e] = eig(XgramG,"vector");
    [XgramB_v,XgramB_e] = eig(XgramB,"vector");

    [Ygram_v,Ygram_e] = eig(Ygram,"matrix");
    [YgramR_v,YgramR_e] = eig(YgramR,"vector");
    [YgramG_v,YgramG_e] = eig(YgramG,"vector");
    [YgramB_v,YgramB_e] = eig(YgramB,"vector");

    [CompGram_v,CompGram_e] = eig(CompGram,"matrix");
    [CompGramR_v,CompGramR_e] = eig(CompGramR,"vector");
    [CompGramG_v,CompGramG_e] = eig(CompGramG,"vector");
    [CompGramB_v,CompGramB_e] = eig(CompGramB,"vector");

       

end

%% Channel Eigen Superposition
XgsRGBe = [XgramR_e,XgramG_e,XgramB_e];
YgsRGBe = [YgramR_e,YgramG_e,YgramB_e];
CGsRGBe = [CompGramR_e,CompGramG_e,CompGramB_e];

%% Grammian Superposition
XgsRGBeGram = transpose(XgsRGBe) * XgsRGBe;
YgsRGBeGram = transpose(YgsRGBe) * YgsRGBe;
CGsRGBeGram = transpose(CGsRGBe) * CGsRGBe;
    


%% Singular Values of Eigen Superposition
[XgsRGBeGram_v,XgsRGBeGram_e] = eig(XgsRGBeGram,"matrix");
[YgsRGBeGram_v,YgsRGBeGram_e] = eig(YgsRGBeGram,"matrix");
[CGsRGBeGram_v,CGsRGBeGram_e] = eig(CGsRGBeGram,"matrix");


%% Values

ResXYX_e = abs(XYin_e - Xgram_e);
ResXYY_e = abs(XYin_e - Ygram_e);
DiffG_e = abs (Xgram_e - Ygram_e);



%% Vectors

ResXYX_v = abs(XYin_v - Xgram_v);
ResXYY_v = abs(XYin_v - Ygram_v);
ResXY_v = abs(Xgram_v - Ygram_v);

%% RMS Average
rms_XG_e = sqrt(trace(Xgram_e* Xgram_e)/rank(Xgram_e));
rms_YG_e = sqrt(trace(Ygram_e * Ygram_e)/rank(Ygram_e));
rms_XYD_e = sqrt(trace(DiffG_e * DiffG_e)/rank(DiffG_e));
rms_IN_e = sqrt(trace(XYin_e * conj(XYin_e))/rank(XYin_e));
rms_CompGram_e = sqrt(trace(CompGram_e * CompGram_e)/rank(CompGram_e));
rms_XgsRGBeGram_e = sqrt(trace(XgsRGBeGram_e)/rank(XgsRGBeGram_e));
rms_YgsRGBeGram_e = sqrt(trace(YgsRGBeGram_e)/rank(YgsRGBeGram_e));
rms_CGsRGBeGram_e = sqrt(trace(CGsRGBeGram_e)/rank(CGsRGBeGram_e));

% Variance
Xgram_ed = abs(Xgram_e - rms_XG_e);
Ygram_ed = abs(Ygram_e - rms_YG_e);
DiffG_ed = abs(DiffG_e - rms_XYD_e);
XYin_ed = abs(XYin_e - rms_IN_e);
CompGram_ed = abs(CompGram_e - rms_CompGram_e);
XgsRGBeGram_ed = abs(sqrtm(XgsRGBeGram_e) - rms_XgsRGBeGram_e);
YgsRGBeGram_ed = abs(sqrtm(YgsRGBeGram_e) - rms_YgsRGBeGram_e);
CGsRGBeGram_ed = abs(sqrtm(CGsRGBeGram_e) - rms_CGsRGBeGram_e);

rxged = rank(Xgram_ed);
rxge = rank(Xgram_e);
rxgv = rank(Xgram_v);

ryged = rank(Ygram_ed);
ryge = rank(Ygram_e);
rygv = rank(Ygram_v);

rdifed = rank(DiffG_ed);
rdife = rank(DiffG_e);
%rdifv = rank(DiffG_v);

rined = rank(XYin_ed);
rine = rank(XYin_e);
rinv = rank(XYin_v);

TraceGX_e = trace(Xgram_e);
TraceGX_ed = trace(Xgram_ed);

TraceGY_e = trace(Ygram_e);
TraceGY_ed = trace(Ygram_ed);

TraceDiffG_e = trace(DiffG_e);
TraceDiffG_ed = trace(DiffG_ed);

TraceXYin_e = trace(XYin_e); 
TraceXYin_ed = trace(XYin_ed);

TraceCompGram_e = trace(CompGram_e); 
aTraceCompGram_e = trace(CompGram_e);
TraceCompGram_ed = trace(CompGram_ed);

aTraceXgsRGBeGram_e = trace(XgsRGBeGram_e);
a2TraceXgsRGBeGram_e = trace(sqrtm(XgsRGBeGram_e));
TraceXgsRGBeGram_ed = trace(XgsRGBeGram_ed);

aTraceYgsRGBeGram_e = trace(YgsRGBeGram_e);
a2TraceYgsRGBeGram_e = trace(sqrtm(YgsRGBeGram_e));
TraceYgsRGBeGram_ed = trace(YgsRGBeGram_ed);

aTraceCGsRGBeGram_e = trace(CGsRGBeGram_e);
a2TraceCGsRGBeGram_e = trace(sqrtm(CGsRGBeGram_e));
TraceCGsRGBeGram_ed = trace(CGsRGBeGram_ed);



sdev_XG_e = sqrt(trace(Xgram_ed * Xgram_ed)/rank(Xgram_ed));
sdev_YG_e = sqrt(trace(Ygram_ed * Ygram_ed)/rank(Ygram_ed));

sdev_XYD_e = sqrt(trace(DiffG_ed * DiffG_ed)/rank(DiffG_ed));
sdev_XYD_ebar = sqrt(trace(DiffG_ed * conj(DiffG_ed))/rank(DiffG_ed));

sdev_IN_e = sqrt(trace(XYin_ed * XYin_ed)/rank(XYin_ed));
sdev_IN_ebar = sqrt(trace(XYin_ed * conj(XYin_ed))/rank(XYin_ed));

sdev_CompG_e = sqrt(trace(CompGram_ed * CompGram_ed)/rank(CompGram_ed));





pd0 = makedist('Poisson','lambda', rms_XG_e);
pd1 = makedist('Poisson','lambda', rms_YG_e);
pd2 = makedist('Poisson','lambda', rms_XYD_e);
pd3 = makedist('Poisson','lambda', rms_IN_e);
pd4 = makedist('Poisson','lambda', rms_CompGram_e);
pd5 = makedist('Poisson','lambda', rms_XgsRGBeGram_e);
pd6 = makedist('Poisson','lambda', rms_YgsRGBeGram_e);
pd7 = makedist('Poisson','lambda', rms_CGsRGBeGram_e);

figure(10);
plot(pd0);

figure(1);
plot(pd1);

figure(2);
plot(pd2);

figure(3);
plot(pd3);

figure(4);
plot(pd4);

figure(5);
plot(pd5);

figure(6);
plot(pd6);

figure(7);
plot(pd7);


pause(100);


%{ 
A = imread("DSC_0089_200.png");
A = rgb2gray(A);
A = double(A)/255;
[row, col] = size(A);
rows = 1:row;
cols = 1:col;

%%Computing Singular Values:

n = 50; %% number of wanted eigenvectors and eigenvalues (AKA HOW COMPRESSED THE IMAGE IS)
rand_vec = randn(col,1); % random eigenvector guess
u_vec = rand_vec./norm(rand_vec); % make into unit vec
K = A'*A;  "A transpose A" matrix
u_new = (K*u_vec)./norm(K*u_vec); % find closer eigenvector 
% itterate until u_new ~= previous u_new
for i = 1:60
    u_new = (K*u_new)./norm(K*u_new);
end

eigvec1 = u_new; % first eigenvector
eigval1 = norm(K*u_new); % first eigenvalue

% Initialize vectors and matrices 

V = zeros(col,n); % vector for eigenvectors
Eig = zeros(n,1); % vector for eigenvalues
V(:,1) = eigvec1;
Eig(1) = eigval1;

% Gram-Schmidt Orthogonalization Method

for i = 2:n % for every other eigenvector 
    u_new = rand_vec/norm(rand_vec);
    tolerance = 10; % initialize tolerance for while loop

    while tolerance >= 0.001 
    u_old = u_new;
    u_star = K*u_old;
    tot_sum = 0;
        for m = 1:i-1
            new_sum = u_star'*V(:,m);
            new_sum = new_sum*V(:,m);
            tot_sum = tot_sum + new_sum;
        end
    u_new = u_star - tot_sum;
    u_new = u_new/norm(u_new);
    tolerance = norm(u_new-u_old);
    end

    % collect eigenvectors and eigenvalues 
    V(:,i) = u_new;
    Eig(i) = norm(K*u_new);
end

%% The Incomplete SVD Decomposition

I = eye(n); % 50x50 identity matrix
Sigma = sqrt(Eig).*I; % Sigma matrix
U = A*V/Sigma; % U matrix

A_new = U*Sigma*V'; % 
%}

% Plot new image 
%%figure(1)
%%imshow(Xgray_e*X_gray_v);
% Plot pre-compressed image
%%figure(2)
%%imshow(Xgray);
