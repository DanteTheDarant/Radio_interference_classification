clear
tic
[test, test2] = generateCombinedSampling(1, 10, 1, 8);
toc
imagesc(test)
figure()
imagesc(test2)