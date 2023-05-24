clear
samples = 30; %number of samples of each SNR and amp level combi
numScans = 20; %Should not be changed
snr = -6:2:30;
%amp = linspace(1,0.05,5);
amp = 1;

for currentSNR = snr
  
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
    channel = randi([1,12]);
    startChannel = randi(79); %for the scan
    
    output = [];
    while (size(output,1) < numScans)
        pause = randi([10,2000]);
        idle = pause*1e-6;

        payloadSize = randi([1,4])*1024;
        in = randi([0, 1], payloadSize, 1);
        

        ekstraPakker = randi(300);
        if (pause < 500)
            numPakker = 300 + ekstraPakker;
        else
            numPakker = 200 + ekstraPakker;
        end
    
        conf = wlanHTConfig('ChannelBandwidth', 'CBW20', ...
                'NumTransmitAntennas', 1, ...
                'NumSpaceTimeStreams', 1, ...
                'SpatialMapping', 'Hadamard', ...
                'MCS', 6, ...
                'GuardInterval', 'Long', ...
                'ChannelCoding', 'BCC', ...
                'AggregatedMPDU', false, ...
                'RecommendSmoothing', true, ...
                'PSDULength', payloadSize);

        
        [tempOutput, label] = generateHTCSampling(conf,channel,numPakker,in,idle,currentSNR,startChannel,amp);

        size(tempOutput,1)

        output = [output ; tempOutput];

    end
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
