%function err_rates_clean = ParSketch(log_file)

n = 25;
c = 1e3:1e3:6e3;

err_rates_clean = zeros(n,length(c));

parfor k = 1:n
    err_rates_clean(k, :) = ...
        CNNSketchLogged(c, sprintf(['log/parCleanSketch-', num2str(k), '.txt']));
end

figure;
plot(c, err_rates_clean', '.');
hold on;
plot(c, mean(err_rates_clean), 'rd-');
hold off;
fid = fopen('log/sketch_rates_clean-v1.csv', 'a');
fwrite(fid,err_rates_clean);
fclose(fid);

%return;
