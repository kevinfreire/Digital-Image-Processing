%% Project 3-2
size = 255;
img = imread('Fig3.38(a)1.jpg');
scaledImg = scaleImage(size, img);
L_mask = [-1,-1,-1; -1,8,-1; -1,-1,-1];
L_img = mySpatialFilter(scaledImg, L_mask);
img_s = add(img,L_img);
figure;
subplot(1,3,1);imshow(img);title('Original Image','fontsize',14);
subplot(1,3,2);imshow(L_img);title('Masked Image','fontsize',14);
subplot(1,3,3);imshow(img_s);title('Sharpened Image','fontsize',14);

%% Project 3-3
size = 255;
img = imread('Fig0340(a)(dipxe_text).tif');
blur_mask = (1/9)*[1,1,1; 1,1,1; 1,1,1];
blurredImage = mySpatialFilter(img, blur_mask);
mask = subtract(img,blurredImage);
boost_img = add(img,multiply(mask,50));
figure;
subplot(2,2,1);imshow(img);title('Original Image','fontsize',14);
subplot(2,2,2);imshow(blurredImage);title('Blurred Image','fontsize',14);
subplot(2,2,3);imshow(mask);title('Mask','fontsize',14);
subplot(2,2,4);imshow(boost_img);title('Highboost Image','fontsize',14);

%% Project 3-4
clear all;
img = imread('Fig4.41(a)1.jpg');
[M, N] = size(img);
cutoff = [10, 30, 60, 160, 460];                % Cut-off Frequency
Cx = 0.5*M;                                     % Center x-value of filter
Cy = 0.5*N;                                     % Center y-value of filter

shift_img = shift_transform(double(img));       % Shift transform from Project 1
fft_img = fft2(double(shift_img));              % Compute FFT of image

figure;
subplot(1,2,1);imshow(img);title('Original Image','fontsize',14);
subplot(1,2,2);imshow(abs(fft_img),[-12 300000]);title('FFT of Original Image','fontsize',14);

for i=1:5
    LPF = gaussFilter(M, N, cutoff(i), Cx, Cy, 0);     % Gaussian Low Pass Filter
    HPF = gaussFilter(M, N, cutoff(i), Cx, Cy, 1);     % Gaussian High Pass Filter

    lowFiltered_img = fft_img.*LPF;
    highFiltered_img = fft_img.*HPF;

    ifft_lowImg = ifft2(lowFiltered_img);
    ifft_highImg = ifft2(highFiltered_img);
    lowInvShift_img = shift_transform(ifft_lowImg);
    highInvShift_img = shift_transform(ifft_highImg);

    figure;
    subplot(2,3,1);imshow(uint8(abs(lowInvShift_img)));title('Low Pass Filtered Image','fontsize',14);
    subplot(2,3,2);imshow(LPF);title('2-D Gaussian LPF','fontsize',14);
    subplot(2,3,3)
       mesh(0:M-1,0:N-1,LPF)
       axis([ 0 M 0 N 0 1])
       h=gca; 
       get(h,'FontSize') 
       set(h,'FontSize',14)
       title('Gaussian LPF H(f)','fontsize',14)
    subplot(2,3,4);imshow(uint8(abs(highInvShift_img)));title('High Pass Filtered Image','fontsize',14);
    subplot(2,3,5);imshow(HPF);title('2-D Gaussian HPF','fontsize',14);
    subplot(2,3,6);
       mesh(0:M-1,0:N-1,HPF)
       axis([ 0 M 0 N 0 1])
       h=gca; 
       get(h,'FontSize') 
       set(h,'FontSize',14)
       title('Gaussian HPF H(f)','fontsize',14)
end

%% Project Functons

% Project 1-3 functions
function myShift = shift_transform(img)
    [r, c] = size(img);
    myShift = zeros(r, c);
    for i = 1:r
        for j = 1:c
            myShift(i, j) = img(i,j) * (-1)^(i+j);
        end
    end
end

% Project 2-3 functions
% Multiplication function
function img_mult = multiply(img1, img2)
    img_mult = uint8(double(img1) .* double(img2));
end

% Addition function
function img_add = add(img1, img2)
    img_add = uint8(double(img1) + double(img2));
end

% Subtraction function
function img_sub = subtract(img1, img2)
    img_sub = uint8(double(img1) - double(img2));
end

% Image Scaling
function [scaledImg] = scaleImage(size, img)
    f_min = double(img) - double(min(img(:)));
    scaledImg = uint8(size*(f_min./max(f_min(:))));
end

% Project 3-1 Function
% 3X3 Spatial Filter function
function H_img = mySpatialFilter(img, H)
    img = double(img);
    % Adding zero pads around original image
    PI = padarray(img,[1,1],0, 'both');
    [row, col] = size(img);
    % Creating empty filtered Image
    H_img = zeros(row, col);
    for i = 1:row
        for j = 1:col
            H_img(i,j) = H(1,1)* PI(i, j) + H(1,2) *  PI(i,j+1) + H(1,3) * PI(i, j+2) + ...
            H(2,1)* PI(i+1,j) + H(2,2) * PI(i+1, j+1) + H(2,3) * PI(i+1, j+2) + ...
            H(3,1) * PI(i+2, j) + H(3,2) * PI(i+2, j+1) + H(3,3) * PI(i+2, j+2);
        end
    end
    H_img = uint8(H_img);
end

% Project 3-4 Functions
% Gaussian Low Pass and High Pass Filter
% To use the Gaussian LPF the type value is 0 and when using Gaussian HPF 
% type value is 1, the cutoff value is the cutoff frequency and M x N are the
% filter dimensions.  The location of the center of the gaussian filter is
% specified using the values of Cx and Cy for example 0.5*M and 0.5*N is the
% center of the 2-D filter.
function filter = gaussFilter(M, N, cutoff, Cx, Cy, type)
    [X, Y]=meshgrid(0:M-1,0:N-1);
    H = exp(-((X-Cx).^2 +(Y-Cy).^2)./(2*cutoff).^2);
    if type == 0
        filter = H;
    elseif type == 1
        filter = 1 - H;                 % High pass filter = 1-low pass filter
    end
end