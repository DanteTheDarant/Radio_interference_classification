% clear
samples = 60; %number of samples of each SNR and amp level combi
numScans = 20; %Should not be changed
snr = -6:2:30;
%amp = linspace(1,0.05,5);
amp = 0.01;


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
    mode = randi([1,4]);
    wifiChannel = randi([1,12]);                
    [output, label] = generateCombinedSampling(mode, currentSNR, amp, wifiChannel);

    
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