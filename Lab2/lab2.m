%% Project 2-1
img = imread('Fig3.08(a).jpg');
c = 1.5;
figure;
subplot(1,2,1); imshow(img); title('Original Image');
log_img = logTransform(c,img);
subplot(1,2,2); imshow(log_img); title('Log transformation (s = c log(1+r))');
figure;
gamma = 0.3;
power_img = powerLaw_transform(c, gamma, img);
subplot(1,2,1); imshow(img); title('Original Image');
subplot(1,2,2); imshow(power_img); title('Power-Law Transformation (s = cr^\gamma)');

%% Project 2-2
img = imread('Fig3.08(a).jpg');
[H, bins] = myHist(img);
subplot(1,2,1); bar(bins, H); title('Histogram of Original Image');
Eq = myHistEq(H, img);
subplot(1,2,2); bar(bins, Eq); title('Equalized Histogram');
figure;
imgEq = img_Eq(img, Eq);
subplot(1,3,1); imshow(img); title('Original Image');
subplot(1,3,2); imshow(imgEq); title('Equalized Image');
[EH, Ebins] = myHist(imgEq);
subplot(1,3,3); bar(bins,EH); title('Histogram Equalized Image');
%% Project 2-3
img1 = imread('Fig3.08(a).jpg');
img2 = imread('Fig3.38(a)1.jpg');
size = 255;
[scaledImg1, scaledImg2] = scaleImage(size, img1, img2);
figure;
subplot(2,3,1); imshow(img1); title('Original Image 1');
subplot(2,3,4); imshow(img2); title('Original Image 2');
mult_img = multiply(scaledImg1, scaledImg2);
subplot(2,3,2); imshow(mult_img); title('Image multiplication');
div_img = divide(scaledImg1, scaledImg2);
subplot(2,3,3); imshow(div_img); title('Image Division');
add_img = add(scaledImg1, scaledImg2);
subplot(2,3,5); imshow(add_img); title('Image Addition');
sub_img = subtract(scaledImg1, scaledImg2);
subplot(2,3,6); imshow(sub_img); title('Image Subtraction');

%% Project 3-1
img = imread('Fig3.08(a).jpg');
figure;
subplot(2,2,1); imshow(img); title('Original Image');
img = double(img);
H1 = 2*[-1,-1,-1; 0,0,0; 1,1,1];    % Edge detection in the x-direction
H2 = 2*[-1,0,1;-1,0,1;-1,0,1];      % Edge detection in the y-direction
H3 = (1/4)*[0,-1,0;-1,8,-1;0,-1,0]; % Example of a Laplacian filter
H1_img = mySpatialFilter(img, H1);
subplot(2,2,2); imshow(H1_img); title('H1 Spatial Filter');
H2_img = mySpatialFilter(img, H2);
subplot(2,2,3); imshow(H2_img); title('H2 Spatial Filter');
H3_img = mySpatialFilter(img, H3);
subplot(2,2,4); imshow(H3_img); title('H3 Spatial Filter');

%% Project Functions

% Project 2-1 Functions
% log transformation function
function log_img = logTransform(constant, img)
    % Convert image to type double
    r = double(img);
    c=constant;
    % Apply the log transformation
    s = c*log(1+r);
    % Display image range [0 255]
    temp = 255/(c*log(256));
    log_img = uint8(temp*s);
end

% Power-law transformation
function power_img = powerLaw_transform(constant, gamma, img)
    g = gamma; c = constant; r = double(img);
    s = c*(r.^g);
    temp = 255/(c*(255.^g));
    power_img = uint8(temp*s);
end

% Project 2-2 Functions
% Histogram function
function [img_hist, bins] = myHist(img)
    [row, col] = size(img);
    bins = [0:1:255];
    img_hist = zeros(1, 256);
    for i = 1:row
        for j = 1:col
            img_hist(img(i,j)+1) = img_hist(img(i,j)+1) + 1;
        end
    end
end
% Equalized Histogram
function img_Eq = myHistEq(count, img)
    [row, col] = size(img);
    img_Eq = zeros(1, 256);
    numPixels = row*col;
    sum = 0;
    for i = 1:256
        sum = (255 * (count(i)/numPixels)) + sum;
        img_Eq(i) = round(sum);
    end
end
% Enhanced Image with Histogram Equalizer
function enhanced_img = img_Eq(img, histEq)
    [row, col] = size(img);
    enhanced_img = zeros(row, col);
    for i=1:row
        for j=1:col
            enhanced_img(i,j) = histEq(img(i,j)+1);
        end
    end
    enhanced_img = uint8(enhanced_img);
end

% Project 2-3 functions
% Multiplication function
function img_mult = multiply(img1, img2)
    img_mult = uint8(double(img1) .* double(img2));
end

% Division function
function img_div = divide(img1, img2)
    img_div = uint8(double(img1) ./ double(img2));
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
function [scaledImg1, scaledImg2] = scaleImage(size, img1, img2)
    f1_min = double(img1) - double(min(img1(:)));
    f2_min = double(img2) - double(min(img2(:)));
    Img1 = uint8(size*(f1_min./max(f1_min(:))));
    Img2 = uint8(size*(f2_min./max(f2_min(:))));
    scaledImg1 = imresize(Img1, [size size]);
    scaledImg2 = imresize(Img2, [size size]);
end

% Project 3-1 Function
% 3X3 Spatial Filter function
function H_img = mySpatialFilter(img, H)
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