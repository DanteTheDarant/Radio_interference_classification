
close all 
clear

%% Initialize Parameters for Waveform Generation

% Specify the input parameters for generating Bluetooth LE waveform
numPackets = 50;    % Number of packets to generate
sps = 160;           % Samples per symbol
messageLen = 2000;  % Length of message in bits
phyMode = 'LE1M';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};
channelBW = 2e6;    % Channel spacing (Hz) as per standard
numChannels = 39;
numScanChannels = 79;


%Setting up the parameters for frequency hopping
alg = 2;
Hop_increment = 5;
usingChannels = 10;
AccessAdress = 'E89BED68';
snr = 40;

% Parameters from the scanning protocol
t_rssi = 10*1e-6;
timeBetweenRSSI = 625*1e-6;  % 625 us approx
startChannel = randi([1 79],1);

% Define symbol rate based on the PHY mode
if any(strcmp(phyMode,{'LE1M','LE500K','LE125K'}))
    symbolRate = 1e6;
    sps = 160;
else
    symbolRate = 2e6;
    sps = 80;

end

%% Waveform Generation and Visualization

% Loop over the number of packets, generating a Bluetooth LE waveform and
% plotting the waveform spectrum
% rng default;
Fs = symbolRate*sps;
mixedBLUE = [];
Told = 0;
labL_T= [];

%% Setting up parameters for the channel selection algorithm 

csa = bleChannelSelection('Algorithm', alg);

if (alg==1)
    csa.HopIncrement = Hop_increment;
    % csa.UsedChannels = [0, 4, 7, 8, 24, 36]
    csa.UsedChannels = randi([1, 36], 1, usingChannels)
elseif (alg==2)
    csa.AccessAddress = AccessAdress ;
    csa.UsedChannels = randi([1, 36], 1, usingChannels)
end

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
 

    Tmax = (length(waveform)+length(mixedBLUE))/Fs;
    Fmix = channelIndex*channelBW;
    t = (Told:1/Fs:(Tmax-1/Fs))';
%     t = (0:1/Fs:0.01)';
%     current_t = (((i-1)*Tsplit):1/Fs:(i*Tsplit)-(1/Fs))';

    mixerCarr = 1*cos(2*pi*Fmix*t);
%     mixerCarr = mixerCarr(1:length(waveform));
    mixedBLUE = [mixedBLUE ; awgn(waveform.*mixerCarr, snr, 'measured')];
    Told = Tmax;


end

L = length(mixedBLUE);

Y = fft(real(mixedBLUE));

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
% Fs = symbolRate*sps;

f = Fs*(0:(L/2))/L;
plot(f,P1) 
title("Single-Sided Amplitude Spectrum of X(t)")
xlabel("f (Hz)")
ylabel("|P1(f)|")

%% Channelizer

tic


Tmax = length(waveform)/Fs;
Tsplit = 0.1;
%Splitting the signal depending on the total length
totalSplits = ceil(Tmax / Tsplit); 
Tsplit = Tmax/totalSplits;
samplesInSplit = floor(length(mixedBLUE)/totalSplits);

decimationFactor = 80;
separator = dsp.Channelizer(160,decimationFactor);
newFs = Fs/decimationFactor;

separatedSignals = [];
for i = 1:totalSplits
    %currentSignal = mixedWLAN(((i-1)*samplesInSplit)+1:i*samplesInSplit);
    currentSignal = mixedBLUE(1:samplesInSplit);
    mixedBLUE = mixedBLUE(samplesInSplit+1:end);
    separatedPartSignal = separator(currentSignal);

    separatedSignals = [separatedSignals ; separatedPartSignal(:,2:numScanChannels+1);];
    %separatedSignals(((i-1)*samplesInNewSplit)+1:i*samplesInNewSplit,:) = separatedPartSignal(:,2:numChannels+1); %here we discard the first channel
end
    
clear mixedBLUE currentSignal separatedPartSignal
% fvtool(separator,(1:160),'Fs',Fs) %if we want to see the filters
fprintf("Channelisation: ")
toc


%% Zero-padding and rssi
tic
samples_in_rssi = t_rssi / (1/newFs);


samplesBetweenRSSI = timeBetweenRSSI / (1/newFs);

sampleOffset = randi([1 samplesBetweenRSSI],1);

signalLength = length(separatedSignals);
signalLength = signalLength-sampleOffset - mod(signalLength,samplesBetweenRSSI);
timeIntervals = sampleOffset:samplesBetweenRSSI:signalLength+sampleOffset;

outputLength = length(timeIntervals);

outputMatrix = zeros(1,outputLength);
currentChannel = startChannel;
for outputIter = 1:outputLength
    sum = 0;
    for sampleIter = 0:samples_in_rssi-1 %Pulls out the 20 samples at the current time and calculates RSSI
        iSquare = real(separatedSignals(timeIntervals(outputIter)+sampleIter,currentChannel))^2; 
        qSquare = imag(separatedSignals(timeIntervals(outputIter)+sampleIter,currentChannel))^2;
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
fprintf("Resampling: ")
imagesc(outputMatrix)



%% Generate labels

% fprintf("Labeling: ")

labels = zeros(1,numScanChannels);

startLabel = channelIndex; %minus 2 because we ignored the first channel 
for i = startLabel:startLabel+symbolRate/1e6 - 1 % minus 10 because we need it to be centered
    labels(i) = 1;
end
labL_T = horzcat(labL_T, labels); 


