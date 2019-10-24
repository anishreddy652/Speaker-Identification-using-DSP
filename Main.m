dataDir = HelperAN4Download; % Path to data directory, DOwnload datset
%dataset of 5 male and 5 female speakers.
ads = audioDatastore(dataDir, 'IncludeSubfolders',true, ...
    'FileExtensions','.flac', ...
    'LabelSource','foldernames')
[trainDatastore, testDatastore] = splitEachLabel(ads,0.80);%split dataset for training and testing.
trainDatastore
trainDatastoreCount = countEachLabel(trainDatastore)
[sampleTrain, info] = read(trainDatastore);
sound(sampleTrain,info.SampleRate)
reset(trainDatastore)
lenDataTrain = length(trainDatastore.Files);
features = cell(lenDataTrain,1);
for i = 1:lenDataTrain
    [dataTrain, infoTrain] = read(trainDatastore);
    features{i} = HelperComputePitchAndMFCC(dataTrain,infoTrain);
end
features = vertcat(features{:});
features = rmmissing(features);
head(features)   % Display the first few rows
featureVectors = features{:,2:15};

m = mean(featureVectors);
s = std(featureVectors);
features{:,2:15} = (featureVectors-m)./s;
head(features)   % Display the first few rows


inputTable     = features;
predictorNames = features.Properties.VariableNames;
predictors     = inputTable(:, predictorNames(2:15));
response       = inputTable.Label;

trainedClassifier = fitcknn( ...
    predictors, ...
    response, ...
    'Distance','euclidean', ...
    'NumNeighbors',5, ...
    'DistanceWeight','squaredinverse', ...
    'Standardize',false, ...
    'ClassNames',unique(response));

k = 5;
group = response;
c = cvpartition(group,'KFold',k); % 5-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',c);

validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

validationPredictions = kfoldPredict(partitionedModel);
figure
cm = confusionchart(features.Label,validationPredictions,'title','Validation Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

lenDataTest = length(testDatastore.Files);
featuresTest = cell(lenDataTest,1);
for i = 1:lenDataTest
  [dataTest, infoTest] = read(testDatastore);
  featuresTest{i} = HelperComputePitchAndMFCC(dataTest,infoTest);
end
featuresTest = vertcat(featuresTest{:});
featuresTest = rmmissing(featuresTest);
featuresTest{:,2:15} = (featuresTest{:,2:15}-m)./s;
head(featuresTest)   % Display the first few rows
result = HelperTestKNNClassifier(trainedClassifier, featuresTest)
