X_int = imread('squrae.jpg');
X_double = double(X_int);

gray = X_double(:,:,1)/3.0 + X_double(:,:,2)/3.0 + X_double(:,:,3)/3.0;

