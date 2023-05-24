clear

numPackets = 985;    % Number of packets to generate
sps = 160;           % Samples per symbol
messageLen = 1000;  % Length of message in bits
phyMode = 'LE1M';   % Select one mode from the set {'LE1M','LE2M','LE500K','LE125K'};



%Setting up the parameters for frequency hopping
alg = 2;
Hop_increment = 5;
usingChannels = 15;
AccessAdress = 'E89BED68';
snr = 40;
mixAmp = 5;


% hold on
tic
[output,labels] = BLE_gen_optimised(phyMode,numPackets,alg,Hop_increment, usingChannels, AccessAdress, snr,mixAmp, messageLen);
toc
figure
subplot(2,1,1)
imagesc(output)
subplot(2,1,2)
imagesc(labels)


% hold off

