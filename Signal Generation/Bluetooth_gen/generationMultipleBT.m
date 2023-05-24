clear
samples = 50; %number of samples of each SNR and amp level combi
numScans = 20; %Should not be changed
snr = -6:2:30;
%amp = linspace(1,0.05,5);
amp = 1;


%coresUsed = 2;
%parpool('Processes',coresUsed)

for inter = 1:length(snr)
    currentSNR = snr(inter);
    outputString = 'data/' + string(currentSNR) + 'dB_amp' +string(amp*100) + '.csv';
    outputString = strrep(outputString,'-','minus');
    labelString = 'labels/' + string(currentSNR) + 'dB_amp' +string(amp*100) + '_labels.csv';
    labelString = strrep(labelString,'-','minus');

    datatable = readtable(outputString, 'ReadVariableNames', false);  %or true if there is a header
    currentAmountOfSamples = height(datatable);
    %clear datatable
    missingSamples = samples - currentAmountOfSamples;
    printState = string(currentSNR) + 'dB is missing: ' + string(missingSamples)
    if (missingSamples <= 0)
        
    else
        



% outputString = string(snr) + 'dB_amp' +string(amp*100) + '.csv';
% labelString = string(snr) + 'dB_amp' +string(amp*100) + '_labels.csv';
%% GENERATION

for i = 1:missingSamples
    tic
    AccessAdress = 'E89BED68';
    usingChannels = randi([5,15]);
    Hop_increment = randi([5,16]);
    mode = randi([1,4]);
    %message_length_mode = randi([1,2]);
    alg = randi([0,2]);
    if alg < 1
        alg = 1;
    end
    if mode == 1
        phyMode = 'LE2M';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
        numPackets = 1000;    % Number of packets to generate
        messageLen = 2000;  % Length of message in bits
    elseif mode == 2 
        phyMode = 'LE1M';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
        numPackets = 985;    % Number of packets to generate
        messageLen = 1000;  % Length of message in bits
    elseif mode == 3 
        phyMode = 'LE500K';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
        numPackets = 728;    % Number of packets to generate
        messageLen = 512;  % Length of message in bits
    elseif mode == 4 
        phyMode = 'LE125K';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
        numPackets = 547;    % Number of packets to generate
        messageLen = 184;  % Length of message in bits
    end

    mode
    alg
                
    [output, label] = BLE_gen_optimised(phyMode,numPackets,alg,Hop_increment, usingChannels, AccessAdress, currentSNR,amp,messageLen);

    
    while (size(output,1) > numScans)
        output(end,:) = [];
    end
    output = reshape(output',1,numScans*79);
    writematrix(output,outputString,'WriteMode','Append');
    writematrix(label,labelString,'WriteMode','Append');
    toc
    i
end
    
    end
end
