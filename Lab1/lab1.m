%% Project 1-1
part1 = imread('Fig2.21(a).jpg');
figure;
for bits = 1:6
    reduced_img = reduceGrayLevels(bits, part1);
    subplot(2, 3, bits);
    imshow(reduced_img, [0 255]);
    title(['Grey level resolution 2^',num2str(bits)]);
end
%% Project 1-2
part2 = imread('Fig2.19(a).jpg');
shrink_factor = [1/2, 1/4, 1/8, 1/16];
zoom_factor = [2, 4, 8, 16];
[row, col] = size(shrink_factor);
figure;
% Shrinking images and retrieving it.
for i = 1:col
    shrink_img = pixelReplication(shrink_factor(i), part2);
    subplot(2, 2, i);
    imshow(shrink_img);
    title(['Shrink factor = ',num2str(shrink_factor(i))]);
end
% Zooming images and retrieving it.
figure;
for i = 1:col
    shrink_img = pixelReplication(shrink_factor(i), part2);
    zoom_img = pixelReplication(zoom_factor(i), shrink_img);
    subplot(2, 2, i);
    imshow(zoom_img);
    title(['Zoom factor = ',num2str(zoom_factor(i))]);
end
%% Project 1-3
part3 = imread('Fig2.19(a).jpg');
[row, col] = size(part3);
%img = im2double(part3);
figure;
subplot(2, 4, 1); imshow(img); title('Original Image');

shift_img = shift_transform(img);
subplot(2, 4, 2); imshow(shift_img); title('Pre processed image for calculating DFT');

fft_img = fft2(shift_img);
fft_spectrum =  20 * log(abs(fft_img));
subplot(2, 4, 3); imshow(fft_spectrum, [0 128]); title('Spectrum of 2D DFT pre processed image');

filter = LPF(row, col, 0.2);
subplot(2, 4, 4); imshow(filter);title('Low Pass Filter Mask');

fft_filter = fft2(filter);
LP_fft = fft_img.*filter;
subplot(2, 4, 5); imshow(LP_fft);title('Low passed output');

ifft_img = ifft2(LP_fft);
subplot(2, 4, 6); imshow(ifft_img);title('Output image after inverse 2D DFT');

shift_ifft = shift_transform(ifft_img); 
spectrum = 1 + log(abs(shift_ifft));
subplot(2, 4, 7); imshow(spectrum); title('Post Processed Image');
%% Functions for Project

%Project 1-1 function
function reduced_img = reduceGrayLevels(bits, img)
    target_levels = 2^bits;
    comp_factor = 256./target_levels;
    % This if statement is placed in order to retreive an image with either
    % a 0 gray level or 256 gray level
    if bits == 1
        comp_factor = 256
    end
    reduced_img = uint8(floor(double(img)/256 * target_levels) * comp_factor);
end

% Project 1-2 function
function resized_img = pixelReplication(factor, img)
    [W,L] = size(img);
    resized_img = uint8(zeros(W*factor,L*factor));
    if factor < 1
        row_count = 1;
        for row = 1:(1/factor):W
            col_count = 1;
            for col = 1:(1/factor):L
                resized_img(row_count, col_count) = img(row, col);
                col_count = col_count + 1;
            end
            row_count = row_count + 1;
        end
    else
        row_count = 1;
        for row = 1:W
        col_count = 1;
            for col = 1:L
                for i = 1:factor
                    for j = 1:factor
                        resized_img(row_count+i, col_count+j) = img(row, col);
                    end
                end
                col_count = col_count + factor;
            end
            row_count = row_count + factor;
        end
    end
end

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

function filter = LPF(row, col, cut_off)
    % Generating a Real, Symmetric Filter Function
    % Implement a "Low Pass Filter" using "freqspace" matlab command
    [x,y] = freqspace([row col], 'meshgrid');
    z = zeros(row,col);
    for i = 1:row
        for j = 1:col
            z(i,j) = sqrt(x(i,j).^2 + y(i,j).^2);
        end
    end
    % Choosing the Cut off Frequency and defining the low pass filter mask
    filter = zeros(row,col);
    for i = 1:row
        for j = 1:col
            if z(i,j) <= cut_off  % The cut-off frequency of the LPF
                filter(i,j) = 1;
            else
                filter(i,j) = 0;
            end
        end
    end
end