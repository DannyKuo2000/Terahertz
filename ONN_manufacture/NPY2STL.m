clear;
clc;

dx = 0.075;
dh = 0.0025;

input_file = fullfile('npy_output', 'test_261030_1.npy');
output_dir = 'stl_output';
output_file = fullfile(output_dir, 'test_261030_1.stl');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

im = readNPY(input_file);

size_ = size(im);

x0 = linspace(-size_(1)/2, (size_(1)+1)/2, size_(1)) .* dx;
y0 = linspace(-size_(2)/2, (size_(2)+1)/2, size_(2)) .* dx;
z0 = linspace(-size_(3)/2, (size_(3)+1)/2, size_(3)) .* dh;

CONVERT_voxels_to_stl(output_file, im, x0, y0, z0, 'ascii');

fprintf('STL saved to: %s\n', output_file);
