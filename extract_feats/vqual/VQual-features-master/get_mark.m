function marks = get_mark(isValidFrame, frameShift)

% isValidFrame: a vector, 1 for valid, 2 for invalid
% frameShift:   a scalar [sec]



diff = isValidFrame(2:end) - isValidFrame(1:end-1);

marks = [];
for i = 1:length(diff)
    
    if diff(i) == 1
        pStart = i;
    elseif diff(i) == -1
        pEnd   = i;
        
        tmp = [pStart pEnd-pStart];
        marks = [marks; tmp];
    end
    
end

marks = marks * frameShift;

end