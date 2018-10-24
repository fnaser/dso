close all, clear all, clc;

%% READEME

% 1) plot point cloud
% 2) select 3 points on the ground plane
% 3) plot point cloud and plane
% 4) select 4 points for the ROI
% 5) export points to txt for seq

%% Load data

pts = csvread('./example.csv');
pc = pointCloud(pts');

%% tum seq 28
P1 = [-1.5977 -2.8942 4.7679];
P2 = [-3.2313 -2.9793 3.1786];
P3 = [0.3451 0.4708 1.1789];

%% drl seq test 1
P1 = [0.9886 0.7224 1.2376];
P2 = [3.6296 0.6955 0.9409];
P3 = [2.2933 1.8033 -7.5879];

roi1 = [2.6256 0.7078 1.0375];
roi2 = [2.6014 0.7844 0.4345];
roi3 = [3.2091 0.7728 0.4087];
roi4 = [3.3199 0.6802 1.1222];

[X,Y,Z] = getPlane(P1, P2, P3);

%% plotting

clc, close all;

lw = 3;
delta = 10;

figure
hold on;

pcshow(pc, 'MarkerSize', 10) % pc
plotPoints3D(P1, P2, P3, lw) % points

surf(X,Y,Z) % plane
alpha 0.01

% ROI
plot3(roi1(1), roi1(2), roi1(3), 'b*','LineWidth', lw)
plot3(roi2(1), roi2(2), roi2(3), 'ro','LineWidth', lw)
plot3(roi3(1), roi3(2), roi3(3), 'y*','LineWidth', lw)
plot3(roi4(1), roi4(2), roi4(3), 'gx','LineWidth', lw)

xlabel("X")
ylabel("Y")
zlabel("Z")
axis([-5 5 -5 5 -5 5])
hold off;

%% Functions

function [X,Y,Z] = getPlane(P1, P2, P3)
v = cross(P1-P2, P1-P3);
vn = v/norm(v);
w = null(vn);
[P,Q] = meshgrid(-5:0.1:5);
X = P1(1)+w(1,1)*P+w(1,2)*Q;
Y = P1(2)+w(2,1)*P+w(2,2)*Q;
Z = P1(3)+w(3,1)*P+w(3,2)*Q;
end

function plotPoints3D(P1, P2, P3, lw, symbol)
if ~exist('symbol','var')
    symbol = [ 'bo' ; 'rx' ; 'g*' ];
end
plot3(P1(1), P1(2), P1(3), ...
    symbol(1,:), 'LineWidth', lw, 'markers', 10);
plot3(P2(1), P2(2), P2(3), ...
    symbol(2,:), 'LineWidth', lw, 'markers', 10);
plot3(P3(1), P3(2), P3(3), ...
    symbol(3,:), 'LineWidth', lw, 'markers', 10);
end
