function [yupper, ylower, upper_peaks, lower_peaks] = evaluate_envelope(x, n, m)
% pre-allocate space for results
nx = size(x,1);
yupper = zeros(size(x),'like',x);
ylower = zeros(size(x),'like',x);

% handle default case where not enough input is given
if nx < 2
    yupper = x;
    ylower = x;
    return
end

% compute upper envelope
for chan=1:size(x,2)
    if nx > n+1
        % find local maxima separated by at least N samples
        [~,iPk] = findpeaks(double(x(:,chan)),'MinPeakDistance',n, 'MinPeakProminence', m);
    else
        iPk = [];
    end
    
    if numel(iPk)<2
        % include the first and last points
        iLocs = [1; iPk; nx];
    else
        iLocs = iPk;
    end
    
    % smoothly connect the maxima via a spline.
    yupper(:,chan) = interp1(iLocs,x(iLocs,chan),(1:nx)','pchip');
end
upper_peaks = iPk;
% compute lower envelope
for chan=1:size(x,2)
    if nx > n+1
        % find local minima separated by at least N samples
        [~,iPk] = findpeaks(double(-x(:,chan)),'MinPeakDistance',n, 'MinPeakProminence', m);
    else
        iPk = [];
    end
    
    if numel(iPk)<2
        % include the first and last points
        iLocs = [1; iPk; nx];
    else
        iLocs = iPk;
    end
    
    % smoothly connect the minima via a spline.
    ylower(:,chan) = interp1(iLocs,x(iLocs,chan),(1:nx)','pchip');
end
lower_peaks = iPk;
end