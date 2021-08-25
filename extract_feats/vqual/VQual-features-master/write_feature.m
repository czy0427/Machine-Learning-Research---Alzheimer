
function write_feature(outFilePath, featureName, featureVal, ftrLen)

    
    if ftrLen ~= size(featureVal,1)
        error('feature length inconsistent')
    end
    
    try
        h5create(outFilePath,['/features/' featureName],[ftrLen 1]);
    catch ME
        switch ME.identifier
            case 'MATLAB:imagesci:h5create:datasetAlreadyExists'
                % do nothing
                % just skip creating a new dataset
                % the dataset will be overwritten
            otherwise
                error(ME.message)
        end
    end
    
    h5write(outFilePath, ['/features/' featureName],featureVal);
end