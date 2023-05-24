function [outputMatrix,labels] = generateNonHTCSampling(wlanConfig,wifiChannel,numPackets,payload,idle_time,snr,startChannel,mixAmplitude)
    %% SETUP THINGS
    %clear
    if (wifiChannel > 12)
        exit;
    end
    Fmix = wifiChannel*5e6 + 7e6;
    t_rssi = 10*1e-6;
    timeBetweenRSSI = 625*1e-6;  % 625 us approx
    numChannels = 79;
    %startChannel = randi([1 numChannels],1);
    
    Tsplit = 0.4; %time per split in processing, lower to decrease RAM needed. Higher speeds up the calculations a lot
    
    %% Generating 802.11n/ac (OFDM) waveform
    %tic
    % 802.11n/ac (OFDM) configuration
    nonHTCfg = wlanConfig;
    
    if (nonHTCfg.ChannelBandwidth == 'CBW20')
        oversampling = 8;
        labelWidth = 20;
    elseif (nonHTCfg.ChannelBandwidth == 'CBW40')
        oversampling = 4;
        labelWidth = 40;
    elseif (nonHTCfg.ChannelBandwidth == 'CBW10')
        oversampling = 16;
        labelWidth = 10;
    
    end

    %Generation for noise levels 
    waveform = wlanWaveformGenerator(payload, nonHTCfg, ...
        'NumPackets', 3, ...
        'IdleTime', 0, ...
        'OversamplingFactor', oversampling, ...
        'ScramblerInitialization', 93, ...
        'WindowTransitionTime', 1e-07);
    
    Fs = wlanSampleRate(nonHTCfg, 'OversamplingFactor', oversampling); 
    Tmax = length(waveform)/Fs;
    t = (0:1/Fs:Tmax-(1/Fs))';
    mixerSine = mixAmplitude*cos((2*pi*Fmix*t));

    levelForNoise = pow2db(rms(waveform.*mixerSine)^2); %output in dBm - 30 = dBW which is used in the awgn func
    %levelForNoiseW = pow2db(rms(waveform.*mixerSine))-30; %output in dBm - 30 = dBW which is used in the awgn func
   
    clear waveform mixerSine Tmax t 
    
    % Generation of the signal
    waveform = wlanWaveformGenerator(payload, nonHTCfg, ...
        'NumPackets', numPackets, ...
        'IdleTime', idle_time, ...
        'OversamplingFactor', oversampling, ...
        'ScramblerInitialization', 93, ...
        'WindowTransitionTime', 1e-07);
    
    %fprintf("WLAN generation: ")
    %toc
    %length(waveform) %debug
    %% Generating the simple mixing signal and mix signals
    %tic
    Tmax = length(waveform)/Fs;
    
    %Splitting the signal depending on the total length
    totalSplits = ceil(Tmax / Tsplit); 
    Tsplit = Tmax/totalSplits;
    samplesInSplit = floor(length(waveform)/totalSplits);
    
    mixedWLAN = [];
    %t = (0:1/Fs:Tmax-1/Fs)'; %timevector for generating mixer signal
    for i = 1:totalSplits
    %     current_t = t(1:samplesInSplit);
    %     t = t(samplesInSplit+1:end);
        current_t = (((i-1)*Tsplit):1/Fs:(i*Tsplit)-(1/Fs))';
    
        mixerSine = mixAmplitude*cos((2*pi*Fmix*current_t));
    
        current_wave = awgn(waveform(1:samplesInSplit), snr, levelForNoise);
        waveform = waveform(samplesInSplit+1:end);
        
        mixedWLAN = [mixedWLAN ; current_wave.*mixerSine];
        % mixedWLAN(((i-1)*samplesInSplit)+1:i*samplesInSplit) = awgn(current_wave.*mixerSine, specSNR, 'measured'); %OLD code which is faster but requires more memory  
    
    end
    clear current_wave current_t
    
    %mixerSine = mixAmplitude*cos((2*pi*Fmix*t));
    %clear t
    
    %waveform = awgn(waveform, specSNR, 'measured');
    %mixedWLAN = waveform.*mixerSine;
    clear mixerSine waveform 
    
    %adding noise
    %mixedWLAN = awgn(mixedWLAN, specSNR, 'measured');
    
    %fprintf("Mixing and noise addition: ")
    %toc
    
    %% Channeliser
    %tic
    %currently separating the whole signal into 160 signals of the same
    %timelength - could maybe be split in the time domain before channelisation
    %for a speedup (depending on RAM)
    
    decimationFactor = 80;
    separator = dsp.Channelizer(160,decimationFactor);
    newFs = Fs/decimationFactor;
    %samplesInNewSplit = round(samplesInSplit/decimationFactor);
    %separatedSignals = separator(mixedWLAN);
    
    % separatedSignals = zeros(round(length(mixedWLAN)/decimationFactor,numChannels)); %not working currently and will consume a lot of ram
    separatedSignals = [];
    for i = 1:totalSplits
        %currentSignal = mixedWLAN(((i-1)*samplesInSplit)+1:i*samplesInSplit);
        currentSignal = mixedWLAN(1:samplesInSplit);
        mixedWLAN = mixedWLAN(samplesInSplit+1:end);
        separatedPartSignal = separator(currentSignal);
    
        separatedSignals = [separatedSignals ; separatedPartSignal(:,2:numChannels+1);];
        %separatedSignals(((i-1)*samplesInNewSplit)+1:i*samplesInNewSplit,:) = separatedPartSignal(:,2:numChannels+1); %here we discard the first channel
    end
    
    clear mixedWLAN currentSignal separatedPartSignal
    %fvtool(separator,(1:160),'Fs',Fs) %if we want to see the filters
    %fprintf("Channelisation: ")
    %toc
    %% Make zero padding and RSSI part
    %tic
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
    
        if (currentChannel == numChannels)
            currentChannel = 1;
        else
            currentChannel = currentChannel + 1;
        end
    end
    
    
    %Below we remove extra samples so all channels have the same amount of samples
    %Afterwards we reshape it to have channel 1 in column 1 and so on
    
    outputDim = floor(outputLength/numChannels);                    %dimensions for output matrix
    outputMatrix = reshape(outputMatrix(1:outputDim*numChannels),numChannels,outputDim)';    %changes the output to 1 column = 1 channel
    outputMatrix = circshift(outputMatrix,startChannel-1,2);          %ensures that channel one is in column one
    %fprintf("Resampling: ")
    %toc
    
    
    %% Generate labels
    %tic 
    labels = zeros(1,numChannels);
    startLabel = wifiChannel*5 + 7; %minus 2 because we ignored the first channel 
    for i = startLabel-10:startLabel+10 % minus 10 because we need it to be centered
        labels(i) = 1;
    end
    
    
    %fprintf("Labeling: ")
    %toc 
end