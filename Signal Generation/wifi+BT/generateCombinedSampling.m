function [outputMatrix,labels] = generateCombinedSampling(mode, snr, mixAmp, wifiChannel)

%% General parameters
numScanChannels = 79;
decimationFactor = 80;
separator = dsp.Channelizer(160,decimationFactor);
Fs = 160*1e6;     
newFs = Fs/decimationFactor;

% Parameters from the scanning protocol
t_rssi = 10*1e-6;
timeBetweenRSSI = 625*1e-6;  % 625 us approx
startChannel = randi([1 numScanChannels],1); %where we start our scanning 



if mode == 1
    phyMode = 'LE2M';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
    numPackets = 1000;    % Number of packets to generate
    messageLen = 2000;  % Length of message in bits
    waveform_lenght = 163840;
    sps = 80;
elseif mode == 2 
    phyMode = 'LE1M';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
    numPackets = 985;    % Number of packets to generate
    messageLen = 1000;  % Length of message in bits
    waveform_lenght = 166400;
    sps = 160;
elseif mode == 3 
    phyMode = 'LE500K';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
    numPackets = 728;    % Number of packets to generate
    messageLen = 512;  % Length of message in bits
    waveform_lenght = 224960;
    sps = 160;
elseif mode == 4 
    phyMode = 'LE125K';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
    numPackets = 547;    % Number of packets to generate
    messageLen = 184;  % Length of message in bits
    waveform_lenght = 299520;
    sps = 160;
end
dec_wave_length = waveform_lenght/decimationFactor;
min_waveform_lenght = ceil((waveform_lenght*numPackets)/decimationFactor); %the minimum needed waveformlength (stolen from BT generation)



%% Setup of wifi
Fmix = wifiChannel*5e6 + 7e6;
oversampling = 8;
labelWidth = 20;

Tsplit = 0.2; %time per split in processing, lower to decrease RAM needed. Higher speeds up the calculations a lot



%% Find signal levels for adding noise correctly
cfg = wlanHTConfig('ChannelBandwidth', 'CBW20', ...
            'NumTransmitAntennas', 1, ...
            'NumSpaceTimeStreams', 1, ...
            'SpatialMapping', 'Hadamard', ...
            'MCS', 6, ...
            'GuardInterval', 'Long', ...
            'ChannelCoding', 'BCC', ...
            'AggregatedMPDU', false, ...
            'RecommendSmoothing', true, ...
            'PSDULength', 1024);
payload = randi([0, 1], 1024, 1);
waveform = wlanWaveformGenerator(payload, cfg, ...
    'NumPackets', 8, ...
    'IdleTime', 0, ...
    'OversamplingFactor', oversampling, ...
    'ScramblerInitialization', 93, ...
    'WindowTransitionTime', 1e-07);

Tmax = length(waveform)/Fs;
t = (0:1/Fs:Tmax-(1/Fs))';
mixerSine = mixAmp*cos((2*pi*Fmix*t));

levelForNoise = pow2db(rms(waveform.*mixerSine)^2);

clear waveform mixerSine Tmax t payload cfg

lengthIndex = 1;

channelized_wifi = zeros(min_waveform_lenght, numScanChannels);
%% Generating the wifi
while (lengthIndex < min_waveform_lenght)
    pause = randi([10,2000]);
    idle = pause*1e-6;

    ekstraPakker = randi(150);
    if (pause < 500)
        numPakker = 300 + ekstraPakker;
    elseif (pause < 1000)
        numPakker = 200 + ekstraPakker;
    else
        numPakker = 125 + ekstraPakker;
    end

    payloadSize = randi([1,4])*1024;
    wifiPayload = randi([0, 1], payloadSize, 1);

    wifiConf = wlanHTConfig('ChannelBandwidth', 'CBW20', ...
            'NumTransmitAntennas', 1, ...
            'NumSpaceTimeStreams', 1, ...
            'SpatialMapping', 'Hadamard', ...
            'MCS', 6, ...
            'GuardInterval', 'Long', ...
            'ChannelCoding', 'BCC', ...
            'AggregatedMPDU', false, ...
            'RecommendSmoothing', true, ...
            'PSDULength', payloadSize);


    waveform = wlanWaveformGenerator(wifiPayload, wifiConf, ...
    'NumPackets', numPakker, ...
    'IdleTime', idle, ...
    'OversamplingFactor', oversampling, ...
    'ScramblerInitialization', 93, ...
    'WindowTransitionTime', 1e-07);
    

    Tmax = length(waveform)/Fs;
    %Splitting the signal depending on the total length
    totalSplits = ceil(Tmax / Tsplit); 
    Tsplit = Tmax/totalSplits;
    samplesInSplit = floor(length(waveform)/totalSplits);
    
    for i = 1:totalSplits
        current_t = (((i-1)*Tsplit):1/Fs:(i*Tsplit)-(1/Fs))';
        mixerSine = mixAmp*cos((2*pi*Fmix*current_t));
    
        current_wave = waveform(1:samplesInSplit);
        waveform = waveform(samplesInSplit+1:end);
        
        %current_wave = awgn(current_wave.*mixerSine, snr, levelForNoise);
        current_wave = current_wave.*mixerSine;
        separatedPartSignal = separator(current_wave);
        
        sepSignalLen = length(separatedPartSignal);

        if (lengthIndex + sepSignalLen-1) > min_waveform_lenght
            separatedPartSignal = separatedPartSignal(1:sepSignalLen-((lengthIndex + sepSignalLen-1)-min_waveform_lenght),:); 
            sepSignalLen = length(separatedPartSignal);
        end

        channelized_wifi(lengthIndex:lengthIndex+sepSignalLen-1,:) = separatedPartSignal(:,2:numScanChannels+1);
        lengthIndex = lengthIndex + sepSignalLen;
    end
    clear current_wave current_t separatedPartSignal waveform mixerSine


end

%% Generate labels for wifi
%tic 
labels = zeros(1,numScanChannels);
startLabel = wifiChannel*5 + 7; %minus 2 because we ignored the first channel 
for i = startLabel-10:startLabel+10 % minus 10 because we need it to be centered
    labels(i) = 1;
end


%% Generate bluetooth signal    

%% Setting up parameters for the channel selection algorithm 
numberOfChannels = randi([5,15]); 
Hop_increment = randi([5,16]);
csa = bleChannelSelection('Algorithm', 1);
csa.HopIncrement = Hop_increment;
csa.UsedChannels = randperm(36, numberOfChannels); 

%Avoiding channels used by the wifi
wifiCh_in_BT = round((wifiChannel*5 + 7)/2);
avoid_channels = wifiCh_in_BT-5:wifiCh_in_BT+5;
for i=1:numberOfChannels
    while ismember(csa.UsedChannels(i), avoid_channels)
        newChannel = csa.UsedChannels(i);
        while ismember(newChannel, [avoid_channels setdiff(csa.UsedChannels, csa.UsedChannels(i))])
            newChannel = randperm(36, 1); 
        end
        csa.UsedChannels(i) = newChannel;
    end
end


channelBW = 2e6;    % Channel spacing (Hz) as per standard
Told = 0;
mixAmp = mixAmp*0.25; %since bluetooth apparently is slightly more powerfull when generated

%% Generate labels for BT
chan = csa.UsedChannels;

for up=1:length(chan)
    startLabel = 2*chan(up); %minus 2 because we ignored the first channel
    labels(startLabel-1)=2;
    labels(startLabel)=2;
    labels(startLabel+1)=2;
    
end

%% Generate the bluetooth packets
for packetIdx = 1:numPackets
    message = randi([0 1],messageLen,1);    % Message bits generation
%     channelIndex = randi([0 39],1,1);          % Channel index decimal value
    channelIndex = csa();       

    if(channelIndex >=37)
        % Default access address for periodic advertising channels
        accessAddress = [0 1 1 0 1 0 1 1 0 1 1 1 1 1 0 1 1 0 0 ...
                            1 0 0 0 1 0 1 1 1 0 0 0 1]';
    else
        % Random access address for data channels
        % Ideally, this access address value should meet the requirements
        % specified in Section 2.1.2 of volume 6 of the Bluetooth Core
        % Specification.
        accessAddress = [0 0 0 0 0 0 0 1 0 0 1 0 0 ...
            0 1 1 0 1 0 0 0 1 0 1 0 1 1 0 0 1 1 1]';
    end
    waveform = bleWaveformGenerator(message,...
                                        'Mode',phyMode,...
                                        'SamplesPerSymbol',sps,...
                                        'ChannelIndex',channelIndex,...
                                        'AccessAddress',accessAddress);
       
    Tmax = (length(waveform)*packetIdx)/Fs;
    Fmix = channelIndex*channelBW;
    t = (Told:1/Fs:(Tmax-1/Fs))';
    
    mixerCarr = mixAmp*cos(2*pi*Fmix*t);
    waveform = waveform.*mixerCarr;
    test = pow2db(rms(waveform)^2);
    waveform = awgn(waveform, snr, levelForNoise);
    Told = Tmax;

    clear t mixerCarr

    separatedPartSignal = separator(waveform);
    channelized_wifi(1+((packetIdx-1)*dec_wave_length):(packetIdx*dec_wave_length),:) = channelized_wifi(1+((packetIdx-1)*dec_wave_length):(packetIdx*dec_wave_length),:) + separatedPartSignal(:,2:numScanChannels+1);
    
end


%% Make zero padding and RSSI part
%tic
samples_in_rssi = t_rssi / (1/newFs);


samplesBetweenRSSI = timeBetweenRSSI / (1/newFs);

sampleOffset = randi([1 samplesBetweenRSSI],1);

signalLength = length(channelized_wifi);
signalLength = signalLength-sampleOffset - mod(signalLength,samplesBetweenRSSI);
timeIntervals = sampleOffset:samplesBetweenRSSI:signalLength+sampleOffset;

outputLength = length(timeIntervals);

outputMatrix = zeros(1,outputLength);
currentChannel = startChannel;
for outputIter = 1:outputLength
    sum = 0;
    for sampleIter = 0:samples_in_rssi-1 %Pulls out the 20 samples at the current time and calculates RSSI
        iSquare = real(channelized_wifi(timeIntervals(outputIter)+sampleIter,currentChannel))^2; 
        qSquare = imag(channelized_wifi(timeIntervals(outputIter)+sampleIter,currentChannel))^2;
        sum = sum + qSquare + iSquare;
    end
    rssi = 10 * log10(sum/samples_in_rssi);
    outputMatrix(outputIter) = rssi;

    if (currentChannel == numScanChannels)
        currentChannel = 1;
    else
        currentChannel = currentChannel + 1;
    end
end


%Below we remove extra samples so all channels have the same amount of samples
%Afterwards we reshape it to have channel 1 in column 1 and so on

outputDim = floor(outputLength/numScanChannels);                    %dimensions for output matrix
outputMatrix = reshape(outputMatrix(1:outputDim*numScanChannels),numScanChannels,outputDim)';    %changes the output to 1 column = 1 channel
outputMatrix = circshift(outputMatrix,startChannel-1,2);          %ensures that channel one is in column one
%fprintf("Resampling: ")
%toc



end