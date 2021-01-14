%% Project 4-1
img = imread('Fig0503 (original_pattern).tif');
img = double(img);
[M, N] = size(img);
var = 10/256;
m = 50/256;
gauss_noise = gaussianNoise(var, m, M, N);
% histogram(gauss_noise);title('Plot of Gaussian PDF');xlabel('z');ylabel('p(z)');
noisy_img = img + gauss_noise;

% figure;
% histogram(gauss_noise);
A=0.2;
B=0.2;
[Pa, Pb] = impulseNoise(A,B,M,N);
J = uint8(img);
J(Pa) = 0;
J(Pb) = 256;

figure;
subplot(1,2,1);imshow(uint8(img));title('Original Image','fontsize',14)
subplot(1,2,2);imshow(uint8(noisy_img));title('Image with Gaussian Noise','fontsize',14);
figure;
subplot(1,2,1);imshow(uint8(img));title('Original Image','fontsize',14)
subplot(1,2,2);imshow(J);title('Image with Impulse Noise','fontsize',14)

%% Project 4-2
img = imread('Fig0507(a)(ckt-board-orig).tif');
[M,N] = size(img);
% Adding impulse noise
A = 0.2;
B= 0.2;
[Pa,Pb] = impulseNoise(A,B,M,N);
J = img;
J(Pa) = 0;
J(Pb) = 256;

% Filtering image using Median Filter
H = [1,1,1;1,1,1;1,1,1];
filtered_img = mySpatialFilter(J,H);

figure;
subplot(1,3,1);imshow(img);title('Original Image','fontsize',14);
subplot(1,3,2);imshow(J);title('Image with Impulse Noise','fontsize',14);
subplot(1,3,3);imshow(filtered_img);title('Restored Image Using Median Filter','fontsize',14);

%% Project 4-3
img = imread('Fig0526(a)(original_DIP).tif');
[M, N] = size(img);
a = 0.1;
b = 0.1;
T =1;

shift_img = shift_transform(double(img));       % Shift transform from Project 1
fft_img = fft2(double(shift_img));              % Compute FFT of image
fft_spectrum = 20*log(abs(fft_img));

% Adding Blurr to the original Image
blurr = blurrFilter(M, N, T, a, b);
blurr(isnan(blurr))=0;
blurr(isinf(blurr))=0;
blurr_spectrum = 5000*abs(blurr);

filtered_img = fft_img.*blurr;

blurred_ifft = ifft2(filtered_img);

blurred_img = shift_transform(blurred_ifft);
spectrum = abs(blurred_img);

% Adding Gaussian Noise to the blurred image with a variance of 10 and mean
% of 0.
var = 10/256;
m = 0;
gauss_noise = gaussianNoise(var, m, M, N);
blurred_noise = blurred_img + gauss_noise;

% Restoring the image
% Converting degraded image to frequency response
degraded_imgShift = shift_transform(blurred_noise);
degraded_img = fft2(degraded_imgShift);

% Convert Gaussian noise from spatial to frequency domain
gauss = fft2(gauss_noise);

wiener_filter = wienerFilter(fft_img, blurr, gauss);
wiener_filter(isnan(wiener_filter))=0;
wiener_filter(isinf(wiener_filter))=100000;
wiener_spectrum = abs(wiener_filter);

restore = degraded_img.*wiener_filter;

iRestored = ifft2(restore);
restored_img = shift_transform(iRestored);
restored_spectrum = abs(restored_img);

% Plotting different stages of original image
figure;
subplot(2,2,1);imshow(img);title('Original Image','fontsize',14);
subplot(2,2,2);imshow(uint8(spectrum));title('Blurred Image','fontsize',14);
subplot(2,2,3);imshow(uint8(blurred_noise));title('Blurred + Noisy Image','fontsize',14);
subplot(2,2,4);imshow(uint8(restored_spectrum));title('Restored Image','fontsize',14);

% Plotting the filters
figure;
subplot(1,3,1);imshow(uint8(blurr_spectrum));title('Blurr Filter','fontsize',14);
subplot(1,3,2);imshow(uint8(abs(gauss_noise)));title('Gaussian Noise','fontsize',14);
subplot(1,3,3);imshow(wiener_spectrum);title('Wiener Filter','fontsize',14);

%% Project Functions
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

% Project 4-1 Function
function noise = gaussianNoise(std, mean, M, N)
    noise = 256*(sqrt(std)*randn(M,N)+mean);
end

function [Pa,Pb] = impulseNoise(A,B,M,N)
    x = rand(M,N);
    y = rand(M,N);
    Pa = find(x <= A/2);
    Pb = find(y <= B/2);
end

% Project 4-2 Function
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
            H_img(i,j) = median([(H(1,1)*PI(i, j))  (H(1,2)*PI(i,j+1))  (H(1,3)*PI(i, j+2)) ...
            (H(2,1)*PI(i+1,j))  (H(2,2)*PI(i+1, j+1))  (H(2,3)*PI(i+1, j+2)) ...
            (H(3,1)*PI(i+2, j))  (H(3,2)*PI(i+2, j+1))  (H(3,3)*PI(i+2, j+2))]);
        end
    end
    H_img = uint8(H_img);
end

% Blurr Filter Function
function filter = blurrFilter(M, N, T, a, b)
    [X, Y]=meshgrid(1:M,1:N);
    filter = (T./(pi*((X-M/2)*a+(Y-N/2)*b))).*sin(pi*((X-M/2)*a+(Y-N/2)*b)).*exp(-1i*pi*((X-M/2)*a+(Y-N/2)*b));
end

% Wiener Filter Function
function filter = wienerFilter(F, H, N)
    H_inv = 1./H;
    H_abs = conj(H).*H;
    S_n = (abs(N)).^2;
    S_f = (abs(F)).^2;
    filter = H_inv.*(H_abs./(H_abs + (S_n./S_f)));
end