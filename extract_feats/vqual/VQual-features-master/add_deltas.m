function add_deltas( outFilePath, DELTA_WINDOW)
% Add time-derivatives of the extracted features

hInfo  = h5info(outFilePath, '/features/');

NumFtrs = length(hInfo.Datasets);
for i=1:NumFtrs
    
    ftrName = hInfo.Datasets(i).Name;
    
    if ftrName(1) == 'd'
        % ignore time derivatives
        continue;
    end

    ftr = h5read(outFilePath, ['/features/' ftrName]);
    ftrLen = size(ftr,1);

    
    d0ftr = ftr.';        
    d1ftr = get_delta(d0ftr, DELTA_WINDOW);
    d2ftr = get_delta(d1ftr, DELTA_WINDOW);
    
    
    write_feature(outFilePath, ['d' ftrName],  d1ftr.', ftrLen);
    write_feature(outFilePath, ['dd' ftrName], d2ftr.', ftrLen);
end

end
