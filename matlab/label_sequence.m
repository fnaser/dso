close all, clear all, clc;

%% READEME

% 1) plot point cloud
% 2) select 3 points on the ground plane
% 3) plot point cloud and plane
% 4) select 4 points for the ROI
% 5) export points to txt for seq

% P1 = ;
% P2 = ;
% P3 = ;
% roi1 = ;
% roi2 = ;
% roi3 = ;
% roi4 = ;

%% Load data

root = '/home/fnaser/DropboxMIT/Holodeck/ShadowCam_DataSet_Oct2018/sequence_m_2/';

pts = csvread([root 'point_cloud.csv']);
pc = pointCloud(pts);

%% tum seq 28
P1 = [-1.5977 -2.8942 4.7679];
P2 = [-3.2313 -2.9793 3.1786];
P3 = [0.3451 0.4708 1.1789];

%% drl seq test 1
P1 = [0.9886 0.7224 1.2376];
P2 = [3.6296 0.6955 0.9409];
P3 = [2.2933 1.8033 -7.5879];

roi2 = [2.6256 0.7078 1.0375];
roi3 = [2.6014 0.7844 0.4345];
roi4 = [3.2091 0.7728 0.4087];
roi1 = [3.3199 0.6802 1.1222];

%% drl seq m 1
P1 = [0.5799 0.4434 0.7751];
P2 = [-1.6867 1.2422 1.5390];
P3 = [-3.3954 1.8381 1.0850];

% roi1 = [1.2388 0.2107 0.4722];
% roi2 = [1.5242 0.1131 0.8710];
% roi3 = [1.1484 0.2466 1.1726];
% roi4 = [0.6743 0.4103 0.7747];

roi1 = [-6.7832 2.9070 3.3127];
roi2 = [-7.2427 3.1954 2.6090];
roi3 = [-6.4901 2.9292 2.2057];
roi4 = [-6.1085 2.7998 2.9040];

%% drl seq nom 1

P1 = [-1.7149 0.5683 1.0878];
P2 = [-5.7337 0.2847 0.9669];
P3 = [-8.3723 0.3138 2.9823];

% roi1 = [-4.2751 0.3498 0.6430];
% roi2 = [-4.1378 0.4221 1.2562];
% roi3 = [-4.7639 0.4046 1.4963];
% roi4 = [-4.9175 0.2823 0.8632];

% roi1 = [-9.2738 0.1998 2.4651];
% roi2 = [-9.5104 0.1199 1.8425];
% roi3 = [-8.9835 0.1298 1.5929];
% roi4 = [-8.7470 0.2097 2.2155];

roi1 = [-8.0688 0.1645 2.1541];
roi2 = [-8.3574 0.1473 1.3527];
roi3 = [-7.5226 0.1707 1.0325];
roi4 = [-7.1181 0.2945 1.9712];

%% plotting

clc, close all;

[X,Y,Z] = getPlane(P1, P2, P3);

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
axis([-15 15 -15 15 -15 15])
hold off;

%% Functions

function [X,Y,Z] = getPlane(P1, P2, P3)
v = cross(P1-P2, P1-P3);
vn = v/norm(v);
w = null(vn);
[P,Q] = meshgrid(-15:0.1:15);
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
