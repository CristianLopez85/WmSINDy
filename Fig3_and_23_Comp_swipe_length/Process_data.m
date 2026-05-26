clear;clc;close all

% Initialize cell arrays to store data
E_noise = zeros(50, 9, 8);
E_parameter = zeros(50, 9, 8);
E_pre = zeros(50, 9, 8);
E_short_traj = zeros(50, 9, 8);
E_vector_field = zeros(50, 9, 8);
Rate_success = zeros(50, 9, 8);
TrainTime = zeros(50, 9, 8);

% List of file names
file_names = {'mS_noAppN_500.mat', 'mS_noAppN_750.mat', 'mS_noAppN_1000.mat', 'mS_noAppN_1250.mat', ...
              'mS_noAppN_1500.mat', 'mS_noAppN_1750.mat', 'mS_noAppN_2000.mat', 'mS_noAppN_2250.mat', ...
              'mS_noAppN_2500.mat'};

% Load and combine data
for i = 1:length(file_names)
    data = load(file_names{i});
    
    % Store each variable in the corresponding 3D array
    E_noise(:,i,:) = data.E_noise;
    E_parameter(:,i,:) = data.E_parameter;
    E_pre(:,i,:) = data.E_pre;
    E_short_traj(:,i,:) = data.E_short;
    E_vector_field(:,i,:) = data.E_vector_field;
    Rate_success(:,i,:) = data.Success;
    TrainTime(:,i,:) = data.TrainTime;
end

% Save combined data to a new .mat file
save('mLorenz_length_Exp50_q3_l02_noAppN.mat', 'E_noise', 'E_parameter', 'E_pre', ...
     'E_short_traj', 'E_vector_field', 'Rate_success', 'TrainTime');

%%
clear;clc;close all

% Initialize cell arrays to store data
E_noise = zeros(50, 9, 8);
E_parameter = zeros(50, 9, 8);
E_pre = zeros(50, 9, 8);
E_short_traj = zeros(50, 9, 8);
E_vector_field = zeros(50, 9, 8);
Rate_success = zeros(50, 9, 8);
TrainTime = zeros(50, 9, 8);

% List of file names
file_names = {'WmS_noAppN_500.mat', 'WmS_noAppN_750.mat', 'WmS_noAppN_1000.mat', 'WmS_noAppN_1250.mat', ...
              'WmS_noAppN_1500.mat', 'WmS_noAppN_1750.mat', 'WmS_noAppN_2000.mat', 'WmS_noAppN_2250.mat', ...
              'WmS_noAppN_2500.mat'};

% Load and combine data
for i = 1:length(file_names)
    data = load(file_names{i});
    
    % Store each variable in the corresponding 3D array
    E_noise(:,i,:) = data.E_noise;
    E_parameter(:,i,:) = data.E_parameter;
    E_pre(:,i,:) = data.E_pre;
    E_short_traj(:,i,:) = data.E_short;
    E_vector_field(:,i,:) = data.E_vector_field;
    Rate_success(:,i,:) = data.Success;
    TrainTime(:,i,:) = data.TrainTime;
end

% Save combined data to a new .mat file
save('WmLorenz_length_Exp50_q3_l02_noAppN.mat', 'E_noise', 'E_parameter', 'E_pre', ...
     'E_short_traj', 'E_vector_field', 'Rate_success', 'TrainTime');

