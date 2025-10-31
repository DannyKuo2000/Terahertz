clear all;
clc;

dx = 0.075;
dh = 0.0025

%%%%%%%%%%%%%%%

for nn = 1:3        
number = int2str(nn)
file_name = 'param_2.npy'
im = readNPY(file_name);

size_ = size(im);
%diffuser_surf = diffuser_surf(1:10,1:10,1:10);

x0 = linspace(-size_(1)/2, (size_(1)+1)/2, size_(1)).*dx;
y0 = linspace(-size_(2)/2, (size_(2)+1)/2, size_(2)).*dx;
z0 = linspace(-size_(3)/2, (size_(3)+1)/2,size_(3)).*dh;
%diffuser_surf = delaunayTriangulation(diffuser_surf);
CONVERT_voxels_to_stl("param_2.stl",im,x0,y0,z0,'ascii')
end

